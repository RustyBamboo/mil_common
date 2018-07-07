#!/usr/bin/env python
from __future__ import division

import numpy as np

import rospy
import tf2_ros
from tf import transformations
from multilateration import Multilaterator, ls_line_intersection3d, get_time_delta

import threading
import serial
import os

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from mil_passive_sonar.srv import *

__author__ = 'David Soto'

# GLOBALS
lock = threading.Lock()  # prevent multiple threads simultaneously using a serial port

def thread_lock(lock):  # decorator
    '''
    Use an existing thread lock prevent a function from being executed by multiple threads at once
    '''
    def lock_thread(function_to_lock):
        def locked_function(*args, **kwargs):
            with lock:
                result = function_to_lock(*args, **kwargs)
        return locked_function
    return lock_thread

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


class _SerialReceiverArray(ReceiverArrayInterface):
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

class _LoggedReceiverArray(ReceiverArrayInterface):
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

class PassiveSonar(object):
    '''
    Passive Sonar Driver: listens for a pinger and uses multilateration to locate it in a
        specified TF frame.

    Args:
    * input_mode - one of the following strings: ['serial', 'log', 'signal_cb']
        'serial'    - get signal input with a data acquistion board via a serial port
        'log'       - get signal input from a .npz file saved to disk
        'signal_cb' - get signal input from user provided function
        For more information on how to use each of these modes, read the wiki page.
    * signal_callback - Optional function used as a source of input if in 'signal_cb' mode

    This driver can work with an arbitrary receiver arrangement and with as many receivers as your
    heart desires (with n>=4).

    Multiple solvers are available for solving the multilateration problem, each with their pro's
    and cons.

    Services:
      * get_pulse_heading:
          Requests signals from the input source and calculates the relative heading to the pulse
      * estimate_pinger_location
          Estimates the position of a stationary pinger as the least_squares intersection of a set
          of 3D lines in a set TF frame.
      * set_frequency
          Set's the frequncy of the pinger that is being listened to.
      * reset_frequency_estimate
          Flushes all of the saved heading observations, and set's the postion estimate to NaN
      * start_logging
          Starts recording all of the received signals to an internal buffer.
      * save_log
          Dumps the buffer of recorded signals to a file.

    An visualization marker will be published to RVIZ for both the heading and the estimated pinger
    location.

    For more information, read the Passive Sonar wiki page of the mil_common repository.
    '''
    def __init__(self, input_mode='serial', custom_input_source=None):

        self.input_mode = input_mode
        self.load_params()
        self.logging = False

        # TODO: update to use ros_alarms

        if self.input_mode == 'log':
            self.input_source = _LoggedReceiverArray(self.input_src_params['log'])

        elif self.input_mode == 'serial':
            self.input_source = _SerialReceiverArray(self.input_src_params['serial'], self.receiver_count)

        elif self.input_mode == 'custom_cb':
            if not issubclass(custom_input_source, ReceiverArrayInterface):
                raise RuntimeError('The custom input source provided ({}) is not derived from {}'
                    .format(type(custom_input_source), 'ReceiverArrayInterface'))
            self.input_source = custom_input_source

        else:
            raise RuntimeError('\'' + self.input_mode + '\' is not a supported input mode')

        self.reset_position_estimate(None)

        self.multilaterator = Multilaterator(self.receiver_locations, self.c, self.method)

        self.plot_pub = rospy.Publisher('/passive_sonar/plot', Image, queue_size=1)
        self.rviz_pub = rospy.Publisher("/passive_sonar/rviz", Marker, queue_size=10)
        self.declare_services()
        rospy.loginfo('Passive sonar driver initialized')

    # Passive Sonar Helpers

    def load_params(self):
        '''
        Loads all the parameters needed for receiving and processing signals from the passive
        sonar board and calculating headings towards an active pinger.

        These parameters are descrived in detail in the Passive Sonar page of the mil_common wiki.
        TODO: copy url here
        '''
        # ROS params expected to be loaded under namespace passive_sonar
        self.required_params = ['receiver_locations', 'method', 'c', 'target_frequency',
                                'sampling_frequency', 'upsampling_factor', 'locating_frame',
                                'receiver_array_frame', 'min_variance', 'observation_buffer_size',
                                'input_timeout']
        self.input_src_params = \
        {
         'serial' : ['port', 'baud', 'tx_request_code', 'tx_start_code', 'read_timeout',
                     'scalar_size', 'signal_size', 'signal_bias', 'locating_frame',
                     'receiver_array_frame'],
         'log'    : ['log_filename' ,'locating_frame', 'receiver_array_frame']
        }

        load = lambda prop: setattr(self, prop, rospy.get_param('passive_sonar/' + prop))
        try:
            [load(x) for x in self.required_params]
        except KeyError as e:
            raise IOError('A required rosparam was not declared: ' + str(e))

        self.receiver_count = len(self.receiver_locations) + 1
        self.receiver_locations = np.array(  # dictionary to numpy array
            [np.array([x['x'], x['y'], x['z']]) for x in self.receiver_locations])

    def declare_services(self):
       '''
       Conveniently declares all the services offered by the driver
       '''
       services = {
                      'get_pulse_heading'        : GetPulseHeading,
                      'estimate_pinger_position' : EstimatePingerPosition,
                      'reset_position_estimate'  : ResetPositionEstimate,
                      'start_logging'            : StartLogging,
                      'save_log'                 : SaveLog,
                      'set_frequency'            : SetFrequency
                  }
       [rospy.Service('passive_sonar/' + s[0], s[1], getattr(self, s[0])) for s in services.items()]

    def log_data(self, signals, trans, rot):
        '''
        Logs the signal and tf data to a file that can then be palyedback and used as input by 
        running the driver with input_mode='log'
        '''
        if self.logging:
            if signals.size == None:
                self.signals_log = self.signals_log.reshape(0, signals.shape[0],
                                                            signals.shape[1])
            self.signals_log = np.stack((self.signals_log, [signals]))
            self.trans_log = np.stack((self.trans_log, [trans]))
            self.rot_log = np.stack((self.rot_log, [rot]))


    def get_dtoa(self, signals):
        '''
        Returns a list of difference in time of arrival measurements for a signal between
        each of the non_reference hydrophones and the single reference hydrophone.

        signals - (self.receiver_count x self.signal_size) numpy array. It is assumed that the
            first row of this array is the reference signal.
        
        returns: list of <self.receiver_count> dtoa measurements in units of microseconds.
        '''
        sampling_T = 1.0 / self.sampling_frequency
        upsamp_T = sampling_T / self.upsampling_factor
        t_max = sampling_T * signals.shape[1]

        t = np.arange(0, t_max, step=sampling_T)
        t_upsamp = np.arange(0, t_max, step=upsamp_T)

        signals_upsamp = np.array([np.interp(t_upsamp, t, x) for x in signals])

        dtoa, cross_corr, t_corr = \
            map(np.array, zip(*[get_time_delta(t_upsamp, non_ref, signals_upsamp[0]) for non_ref \
                                in signals_upsamp[1 : self.receiver_count]]))

        t_corr = t_corr[0]  # should all be the same
        self.visualize_dsp(t_upsamp, signals_upsamp, t_corr, cross_corr, dtoa)

        print "dtoa: {}".format(np.array(dtoa)*1E6)
        return dtoa

    #Passive Sonar Services

    @thread_lock(lock)
    def get_pulse_heading(self, srv):
        '''
        Returns the heading towards an active pinger emmiting at <self.target_frequency>.
        Heading will be a unit vector in hydrophone_array frame
        '''
        success, err_str = True, ''
        signals, p0, R = None, None, None
        try:
            time = rospy.Time.now()

            # Poll input source until it reports it is ready
            self.input_source.input_request()
            while not self.input_source.ready and not rospy.is_shutdown():
                if rospy.Time.now() - time > rospy.Duration(self.input_timeout):
                    raise IOError('Timed out waiting for input source')
                else:
                    rospy.sleep(0.05)
            self.input_source.reset()

            # Gather input from source
            signals, p0, R = self.input_source.get_input()

            # Carry out multilateration to get heading to pinger
            heading = self.multilaterator.get_pulse_location(self.get_dtoa(signals))
            heading = heading / np.linalg.norm(heading)

        except Exception as e:
            rospy.logerr(str(e))
            heading = np.full(3, np.NaN)
            success = False
            err_str = str(e)

        res = GetPulseHeadingResponse(header=Header(stamp=time, frame_id=self.locating_frame),
                                      x=heading[0], y=heading[1], z=heading[2],
                                      success=success, err_str=err_str)

        try:
            # Log input if self.logging == True
            self.log_data(signals, p0, R)

            # Add heaing observation to buffers if the signals are above the variance threshold
            variance = np.var(signals)
            if variance > self.min_variance:
                map_offset = R.dot(heading)
                p1 = p0 + map_offset

                self.visualize_heading(p0, p1, bgra=[1.0, 0, 0, 0.50], length=4.0)

                self.heading_start = np.append(self.heading_start, np.array([p0]), axis=0)
                self.heading_end = np.append(self.heading_end, np.array([p1]), axis=0)
                self.observation_variances = np.append(self.observation_variances, variance)

                # delete softest samples if we have over max_observations
                if len(self.heading_start) >= self.observation_buffer_size:
                    softest_idx = np.argmin(self.observation_variances)
                    self.heading_start = np.delete(self.line_array, softest_idx, axis=0)
                    self.heading_end = np.delete(self.line_array, softest_idx, axis=0)
                    self.observation_variances = np.delete(self.observation_variances,
                                                           softest_idx, axis=0)
        except Exception as e:
            rospy.logwarn(str(e)) # Service should still return

        print "{}\n{}".format(type(res), res)
        return res

    def estimate_pinger_position(self, req):
        '''
        Uses a buffer of prior observations to estimate the position of the pinger as the intersection
        of a set of 3d lines in the least-squares sense.
        '''
        if len(self.heading_start) > 1:
            raise RuntimeError(
                'Not enough heading observations to estimate the pinger position')
        p = ls_line_intersection3d(self.heading_start, self.heading_end)
        p = self.pinger_postion
        self.visualize_pinger_pos_estimate()
        return {'header' : {'stamp' : ros.Time.now(), 'frame_id' : self.locating_frame},
                'num_headings' : len(self.heading_start),
                'x' : p[0], 'y' : p[1], 'z' : p[2]}

    def reset_position_estimate(self, req):
        '''
        Clears all the heading and amplitude buffers and makes the position estimate NaN
        '''
        self.heading_start = np.empty((0, 3), float)
        self.heading_end = np.empty((0, 3), float)
        self.observation_variances = np.empty((0, 0), float)
        self.pinger_position = np.array([np.NaN, np.NaN, np.NaN])
        return {}

    def start_logging(self, req):
        '''
        Enables logging and checks that we have write access to the desired save path.
        Resets the logged data buffers.
        '''
        self.log_filename = req.filename if req.filename.endswith('.npz') else req.filename + '.npz'

        if not os.access(self.log_filename, os.W_OK):
            raise IOError("Unable to write to file: " + self.log_filename)

        self.logging = True
        self.signals_log = None
        self.trans_log = np.array([]).reshape(0, 3)
        self.rot_log = np.array([]).reshape(0, 3, 3)

    def save_log(self, req):
        '''
        Saves the buffers holding signals, and receiver array poses to a compressed .npz
        file. These files can be used as source of input by running the passive sonar
        driver with input_mode='log' and providing filename in a rosparam.
        '''
        try:
            np.savez_compressed(file=self.log_filename, signal=self.signals_log,
                                trans=self.trans_log, rot=self.rot_log)

        except IOError as e: # Couln't save to specified path
            rospy.logerr(str(e))

        finally:
            self.logging = False

    def set_frequency(self, req):
        '''
        Sets the assumed frequency (in absence of noise) of the signals to received by the driver
        '''
        self.target_frequency = req.frequency
        self.heading_start = np.empty((0, 3), float)
        self.heading_end = np.empty((0, 3), float)
        self.sample_variances = np.empty((0, 1), float)
        self.pinger_position = np.array([np.NaN, np.NaN, np.NaN])
        return {}

    # Visualization

    def visualize_dsp(self, t, signals, t_corr, cross_corr, dtoa):
        '''
        Plots the received signals and cross correlations and publishes the image to /passive_sonar/plot
        '''
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plt.plasma()

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
        axes[0].set_title("Recorded Signals (Black is reference)")
        axes[0].set_xlabel('Time (microseconds)')
        axes[1].set_title("Cross-Correlations)")
        axes[1].set_xlabel('Lag (microseconds)')
        plt.annotate('DTOA: {}'.format(dtoa)

        fig.set_size_inches(9.9, 5.4) # Experimentally determined
        fig.set_dpi(400)
        fig.tight_layout(pad=2)

        axes[0].plot(t, signals[0], linewidth=0.75, color='black') # reference
        axes[0].plot(t, signals[1:].T, linewidth=0.75 )
        axes[1].plot(t_corr, cross_corr.T, linewidth=0.75)

        fig.canvas.draw() # render plot
        plot_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        try:
            self.plot_pub.publish(CvBridge().cv2_to_imgmsg(plot_img, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)  # Intentionally absorb CvBridge Errors

    def visualize_pinger_pos_estimate(self, bgra):
        '''
        Publishes a marker to RVIZ representing the last calculated estimate of the position of
        the pinger.

        rgba - list of 3 or 4 floats in the interval [0.0, 1.0] representing the desired color and
            transparency of the marker
        '''
        marker = Marker()
        marker.ns = "passive_sonar-{}".format(self.target_frequency)
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = self.locating_frame
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        bgra = np.clip(bgra, 0.0, 1.0)
        marker.color.b = bgra[0]
        marker.color.g = bgra[1]
        marker.color.r = bgra[2]
        marker.color.a = 1.0 if len(bgra) < 4 else bgra[3]
        marker.pose.position = numpy_to_point(self.pinger_est_position)
        self.rviz_pub.publish(marker)
        print "position: ({p.x[0]:.2f}, {p.y[0]:.2f})".format(p=self.pinger_position)

    def visualize_heading(self, tail, head, bgra, length=1.0):
        '''
        Publishes an arrow marker to RVIZ representing the heading towards the last heard ping.

        tail - 3x1 numpy array
        head - 3x1 numpy array
        lenth - scalar (float, int) desired length of the arrow marker. If None, length will
            be unchanged.
        '''
        head = tail + (head - tail) * length
        head = Point(head[0], head[1], head[2])
        tail = Point(tail[0], tail[1], tail[2])
        marker = Marker()
        marker.ns = "passive_sonar-{}/heading".format(self.target_frequency)
        marker.header.stamp = rospy.Time(0)
        marker.header.frame_id = self.locating_frame
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.points.append(tail)
        marker.points.append(head)
        marker.color.b = bgra[0]
        marker.color.g = bgra[1]
        marker.color.r = bgra[2]
        marker.color.a = 1.0 if len(bgra) < 4 else bgra[3]
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        self.rviz_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("passive_sonar_driver")

    mode = None
    try:
       mode = rospy.get_param('passive_sonar/input_mode')
    except KeyError as e:
       mode = 'serial'
       rospy.logerr('The param passive_sonar/input_mode was not set, defaulting to \
                    \'serial\'. (' + str(e) + ')')

    ping_ping_motherfucker = PassiveSonar(input_mode=mode)
    rospy.spin()

