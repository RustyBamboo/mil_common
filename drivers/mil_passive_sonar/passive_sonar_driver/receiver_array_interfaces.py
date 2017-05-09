#!/usr/bin/env python
from __future__ import division
import rospy
import numpy as np
import tf2_ros
import serial

__author__ == 'David Soto'

'''
This file contains multiple receiver array interfaces for both simulated and real passive
sonar functionality
'''

class ReceiverArrayInterface(object):
    '''
    This class encapsulates the process of acquiring signals from the receiver array
    and acquiring the pose of the receiver array in a world fixed frame.
    '''
    ready = False  # Request would be immediately responded to if made

    def input_request(self):
	'''
	The driver will call this function to request signal and tf data.
	Inheriting classes SHOULD OVERRIDE this method and do the following:
	1) assign signals to self.signals
	2) assign tf translation to self.translation
	3) assign tf rotation to self.rotation
	4) assign True to self.ready
	'''
	pass

    def get_input(self):
	'''
	Returns a tuple with 3 things: signals, translation, rotation

	signals - a numpy array with a row for each receiver, and a column for each element of
	    the recorded signal. The first row is reserved for the reference signal. The remaining
	    rows should be in the same order as define in rosparam /passive_sonar/receiver_locations.
	translation - a 3 element 1D numpy array representing the position of the receiver array
	    in <ref_frame>. The second element should be a
	rotation - 3x3 rotation matrix (numpy array) representing the orientation of the receiver
	    array in <ref_frame>.

	Derived classes SHOULD NOT OVERRIDE this method.
	'''
	return self.signals, self.translation, self.rotation

    def reset(self):
	'''
	Driver will reset state when signals are received, could be used as feedback that signals
	were received.
	'''
	self.ready = False


class _Serial(ReceiverArrayInterface):
    '''
    This is the default serial ReceiverArrayInterface for the passive sonar driver.
    It is used when the keyword arg input_mode='serial' is passed to the passive sonar
    driver constructor.
    '''
    def __init__(self, param_names, num_receivers):
	self.num_receivers = num_receivers

	load = lambda prop: setattr(self, prop, rospy.get_param('passive_sonar/' + prop))
	try:
	    [load(x) for x in param_names]
	except KeyError as e:
	    raise IOError('A required rosparam was not declared: ' + str(e))

	self.tf2_buf = tf2_ros.Buffer()

	try:
	    self.ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=self.read_timeout)
	    self.ser.flushInput()
	except serial.SerialException, e:
	    rospy.err("Sonar serial connection error: " + str(e))
	    raise e

    def input_request(self):
	try:
	    self._request_signals()
	    self._receive_signals()
	    T = self.get_receiver_pose(rospy.Time.now(), self.receiver_array_frame,
				     self.locating_frame)
	    self.translation, self.rotation = T
	    self.ready = True
	except Exception as e:
	    rospy.logerr(str(e))
	    raise e
 
    def _request_signals(self):
	'''
	Request a set of digital signals from a serial port

	serial_port - open instance of serial.Serial
	data_request_code - char or string to sent to receiver board to signal a data request
	tx_start_code - char or string that the receiver board will send  us to signal the
	    start of a transmission
	'''
	self.ser.flushInput()
	readin = None

	# Request raw signal tx until start bit is received
	while readin == None or ord(readin) != ord(self.tx_start_code):
	    self.ser.write(self.tx_request_code)
	    readin = self.ser.read(1)
	    if len(readin) < len(self.tx_request_code):  # serial read timed out
		raise IOError('Timed out waiting for serial response.')

    def _receive_signals(self):
	'''
	Receives a set of 1D signals from a serial port and packs them into a numpy array

	serial_port - port of type serial.Serial from which to read
	signal_size - number of sacalar elements in each 1D signal
	scalar_size - size of each scalar in bytes
	signal_bias - value to be subtracted from each read scalar before packing into array

	returns: 2D numpy array with <num_signals> rows and <signal_size> columns
	'''
	# this doesn't work well, need to fix
	def error_correction(num_str):
	    return num_str #temp
	    output = ''
	    for char in num_str:
		valid = ord(char) > ord('0') and ord(char) < ord('9')
		output += char if valid else '5'
	    return output

	self.signals = np.full((self.num_receivers, self.signal_size), self.signal_bias, dtype=float)

	for channel in range(self.num_receivers):
	    for i in range(self.signal_size):
		while self.ser.inWaiting() < self.scalar_size:
		    rospy.sleep(0.001)

		self.signals[channel, i] = float(error_correction(self.ser.read(self.scalar_size))) \
					   - self.signal_bias

    def get_receiver_pose(self, time, receiver_array_frame, locating_frame):
	'''
	Gets the pose of the receiver array frame w.r.t. the locating_frame
	(usually /map or /world).

	Returns a 3x1 translation and a 3x3 rotation matrix (numpy arrays)
	'''
	try:
	    tfl = tf2_ros.TransformListener(self.tf2_buf)
	    T = self.tf2_buf.lookup_transform(receiver_array_frame, locating_frame, time,
					      timeout=rospy.Duration(0.20))
	    q = T.transform.rotation
	    t = T.transform.translation
	    rot = transformations.quaternion_matrix([q.w, q.x, q.y, q.z])
	    trans = np.array([t.x, t.y, t.z])
	    return trans, rot[:3,:3]
	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
	    rospy.err(str(e))
	    return None

class _Logged(ReceiverArrayInterface):
    '''
    This is the default logged data ReceiverArrayInterface for the passive sonar driver.
    It is used when the keyword arg input_mode='log' is passed to passive sonar driver
    constructor.
    '''

    def __init__(self, param_names):
	load = lambda prop: setattr(self, prop, rospy.get_param('passive_sonar/' + prop))
	try:
	    [load(x) for x in param_names]
	except KeyError as e:
	    raise IOError('A required rosparam was not declared: ' + str(e))
	self.iter_num = 0
	try:
	    self.np_log = np.load(self.log_filename)
	except Exception as e:
	    rospy.logerr('Unable to access log file at path ' + self.log_filename + '. '
			 + str(e))

    def input_request(self):
	try:
	    self.signals = self.np_log['signal'][self.iter_num]
	    self.translation = self.np_log['trans'][self.iter_num]
	    self.rotation = self.np_log['rot'][self.iter_num]
	    self.ready = True
	    self.iter_num += 1
	except IndexError as e:
	    self.ready = False
	    raise StopIteration('The end of the log was reached.')


class _Simulated(ReceiverArrayInterface):
    '''
    This class will create delayed noisy signals simulating the listening to a sinusoidal
    pulse.
    '''
    def __init__(self, param_names):
	load = lambda prop: setattr(self, prop, rospy.get_param('passive_sonar/' + prop))
	try:
	    [load(x) for x in param_names] # loads tf frame ids
	except KeyError as e:
	    raise IOError('A required rosparam was not declared: ' + str(e))

        # We will need to determine Receiver frame pose in map frame and pinger frame in
        # receiver frame
	self.tf2_buf = tf2_ros.Buffer()

    def input_request(self):
        ''' Respond to a request for input (TF + signals) from the ROS driver '''
        self.pinger_pos_gt = get_pinger_ground_truth_position()
        self.map_pose = self.get_map_frame_pose()
        signals = self.make_delayed_signals(self.pinger_position, duration)
        noise = None # TODO
        # add noise to signals here
        self.signals = signals
        self.translation = self.map_pose.translation
        self.rotation = self.map_pose.rotation
        self.ready = True

        
