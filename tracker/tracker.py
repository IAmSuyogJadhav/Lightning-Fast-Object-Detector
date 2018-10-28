from imutils.video import VideoStream, FPS
from imutils import resize
import numpy as np
import cv2
import os
import threading

# TODO:
# 1.Implement the support for running a function over the detections.


class Tracker():
    """
    Tracker(
        self, detector=None, confidence=0.4, tracker='KCF', func=None
        refresh_interval=20, video=None, width=400, window=None
            )

    Creates a `Tracker` object, designed to automatically detect and
    subsequently track the motion of, any single object.

    Parameters
    ----------
    -`detector` : list/tuple, optional
        A 2-tuple in the format (prototxt, caffemodel).
        `prototxt` is the path to the .prototxt file with description
        of network architecture.
        `caffemodel` is the path to the learned Caffe model to be used
        for detecting the object.
        If not provided, lets the user select a ROI by himself.

    -`confidence` : float (0 < confidence < 1), optional
        The threshold confidence level to select a bounding box. Is
        ignored when a detector is not provided.
        A real number between 0 and 1. Defaults to 0.4. Any bounding box
        having confidence less than this value will be rejected.

    -`refresh_interval` : int, optional
        The no. of frames after which to detect the object again and reset the
        bounding box. Defaults to 5. Ignored if a `detector` is not provided.
        Helps the tracker maintain the correct bounding box.
        Set equal to 0 to disable this feature.

    -`tracker` : {'KCF','CSRT','MIL','Boosting','MedianFlow','TLD','MOSSE'},
                optional
        The name of the tracker to use. Defaults to KCF.

    `func`: function, optional
        A function that accepts a single argument, the cropped object found out
        by the tracker. If provided, will be called whenever a succesful
        detection is made.

        *To be implemented

    `video` : str, optional
        If a valid video path is provided, uses the video for detecting
        and tracking the objecting. Otherwise, uses webcam.

    `width` : int, optional
        The width in pixels to which to resize the frame. Used for reducing
        the computation. Lower the width, lower the computation time,
        but also harder for the detector to detect objects in the image.
        Defaults to 400 px.

    `window` : list/tuple, optional
        Format: [window_name, flags]
            `window_name` is the name for the output window.
            `flags` are the relevat flags that can be passed to cv2.namedWindow
        If provided, displays output in a window with given window information.
        Defaults to ["Output", cv2.WINDOW_GUI_NORMAL]

    Returns
    -------
    A Tracker object.
    """

    def __init__(
        self, detector=None, confidence=0.4, tracker='KCF', func=None,
        refresh_interval=20, video=None, width=400, window=None
            ):

        # Initialize the tracker
        self.tracker_name = tracker
        self.tracker = self._initialize_tracker(self.tracker_name)

        # Initialize the video stream
        if video is None:  # If no video file is provided
            print('Using webcam...\n')
            self.vs = VideoStream(src=0).start()
        elif not os.path.exists(video):
            raise Exception(
                f"The specified file '{video}' couldn't be loaded.")
        else:
            self.vs = cv2.VideoCapture(video)

        # Initialize the detector
        if detector is not None:  # If a detector is provided
            try:
                self.detector = cv2.dnn.readNetFromCaffe(*detector)
                self.detector.setPreferableBackend(00000)
            except Exception as e:  # If the model fails to load
                print(f"The detector couldn't be initialized.")
                raise(e)
        else:
            self.detector = None

        # Sanity check for confidence
        if 0. < confidence < 1.:
            self.confidence = confidence
        else:
            raise Exception("Confidence should lie within 0 and 1.")

        # Save the output window options
        if window is None:  # If no window information is provided,
            # Create a window with the default options
            self.window = ["Output", cv2.WINDOW_GUI_NORMAL]
        else:
            self.window = window

        # Initialize other attributes
        self.func = func
        self.initBB = None  # To store bounding box coordinates
        self.updatedBB = None  # To refresh the bounding box
        self.fps = None  # Initialize fps (frames per second) count
        self.frame = None  # For storing the frame to be displayed
        self.frame_copy = None  # For storing the unmodified frame
        self.interval = refresh_interval  # The refresh interval
        self.frame_count = 0  # Initialize the frame counter

        # Some less important attributes
        self.using_webcam = True if video is None else False
        # To track if the update of bounding box is already underway
        self.update_in_progress = False
        self.width = width  # The width to resize the frame to

    def __del__(self):
        """
        Performs a cleanup. Wrapper for Tracker.stop()
        """
        self.stop()

    def _initialize_tracker(self, tracker_name, update=False):
        """
        Tracker._initialize_tracker(self, tracker_name, update=False)

        Initializes and returns a new tracker instance. If `update`
        is `True`, doesn't print out anything. Internal function.
        """
        if tracker_name.lower() == 'boosting':
            if not update:  # Don't print for reinitialized trackers
                print('Using Boosting tracker.')
            return cv2.TrackerBoosting_create()

        elif tracker_name.lower() == 'medianflow':
            if not update:  # Don't print for reinitialized trackers
                print('Using MedianFlow tracker.')
            return cv2.TrackerMedianFlow_create()

        else:
            try:
                tracker = eval(f'cv2.Tracker{tracker_name.upper()}_create()')
                if not update:  # Don't print for reinitialized trackers
                    print(f'Using {tracker_name.upper()} tracker.')
                return tracker
            except AttributeError:
                raise Exception(f"'{tracker_name}' is not a valid tracker.")

    def _get_BB(self, update=False):
        """
        Tracker._get_BB(self, update=False)

        Get the bounding box to be tracked. If `update` = `True`,
        runs in a separate thread and only saves the new
        bounding box coordinates (as the tracker is already in use).
        If a detector was provided earlier, gives the bounding box
        detected by the detector.
        Else, lets the user select a ROI by himself.
        """

        H, W = self.frame.shape[:2]  # Grab the shape of the frame

        if self.detector is not None:  # If a detector is provided
            # Preprocess frame to pass through the detector and create a blob
            blob = cv2.dnn.blobFromImage(
                cv2.resize(self.frame_copy, (300, 300)), 1.,
                (300, 300), (104.0, 177.0, 123.0)
                )
            self.detector.setInput(blob)  # Set the input image for detector

            # Workaround for a cv2 error
            try:
                # Get the detections.
                detections = self.detector.forward()
            except cv2.error:
                print('error')
                # Reset the parameters and return
                self.update_in_progress = False
                self.frame_count = self.interval - 1
                return

            if detections is not None:  # If anything is detected at all
                # The returned detections are sorted according to
                # their confidences
                # Therefore, the first element is the one we want
                newBB = detections[0, 0, 0, :].squeeze()
                newBB_confidence = newBB[2]  # The confidence for newBB

                # Compute (x, y) coordinates for the bounding box
                box = newBB[3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                w = endX - startX  # Width
                h = endY - startY  # Height
                newBB = [startX, startY, w, h]  # The required format

                # Update the bounding box only if it has greater
                # confidence than the self.confidence threshold
                if newBB_confidence >= self.confidence:

                    if update:  # If an update was requested
                        self.updatedBB = newBB  # Store the updated box

                        # Reset update parameters
                        self.frame_count = 0
                        self.update_in_progress = False
                        return

                    else:
                        self.initBB = newBB
                        # Initialize the tracker
                        self.tracker.init(self.frame, tuple(self.initBB))
                        return
                else:
                    if update:
                        # Reset the parameters so that this function is called
                        # again on the next iteration of the main loop.
                        self.update_in_progress = False
                        self.frame_count = self.interval - 1
                        return

            else:  # If nothing is detected
                # Reset the relevant parameters so that this function is called
                # again on the next iteration of the main loop.
                if update:  # If an update was requested
                    self.update_in_progress = False
                    # Set frame count to one less than the interval,
                    # so that, it triggers this function on next iteration
                    self.frame_count = self.interval - 1
                    return
                else:
                    # Set self.initBB to None, so that this function
                    # will be called again on next iteration
                    self.initBB = None
                    return

        # If a detector is not provided, ignore updates
        elif self.detector is None and not update:
            # Put some on-screen instructions to select the ROI
            info = [
                ('ESC', 'Reselect region'),
                ('ENTER or SPACE', "Confirm selection"),
                ('Select region of interest.', ''),
            ]

            for i, (label, value) in enumerate(info):
                text = f"{label}: {value}"
                cv2.putText(
                    self.frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
                )

            # Select ROI and use as our bounding box
            cv2.destroyWindow('Output')
            self.initBB = cv2.selectROI(self.window[0], self.frame,
                                        showCrosshair=False)

            self.frame = self.frame_copy  # To clear the text on the frame

            if update:  # If an update is requested
                # Re-create the tracker
                self.tracker = self._initialize_tracker(self.tracker_name,
                                                        update=True)

            # Finally, initialize the tracker
            self.tracker.init(self.frame, tuple(self.initBB))

    def _run_func(self):
        """
        Tracker._run_func(self)

        Runs the provided function (`func`) over the detected object.

        *To be implemented
        """
        pass

    def start(self):
        """
        Tracker.start(self)

        Start the detection and tracking pipeline. Generates a window
        showing the results of tracking.

        Usage
        -----
        On the generated output window, press following keys
        to perform the function listed next to them.
        +--------------------+----------------------------------------+
        |        Key         |                Function                |
        +--------------------+----------------------------------------+
        |         S          |    Re-initialize the bounding box.     |
        | Q or Esc or ALT+F4 |    Close the window and stop tracking. |
        +--------------------+----------------------------------------+

        """

        # Create the output window
        try:
            cv2.namedWindow(*self.window)
        except Exception as e:
            print('Couldn\'t create window.')
            raise(e)

        while True:
            # print(self.frame_count)  # DEBUG
            # print('Running threads: ', threading.active_count())  # DEBUG

            self.frame = self.vs.read()  # Grab a frame from the video
            self.frame = \
                self.frame[1] if not self.using_webcam else self.frame

            # To reduce the processing time
            self.frame = resize(self.frame, width=self.width)
            H, W = self.frame.shape[:2]  # Height and width, needed later on
            self.frame_copy = self.frame.copy()  # Preserve an original copy

            if self.frame is None:  # Marks the end of the stream
                print('Stream has ended, exiting...')
                break

            if self.initBB is None:  # The bounding box is not initialized.
                self._get_BB(update=False)  # Get the initial bounding box
                self.fps = FPS().start()  # Start recording FPS

            elif self.updatedBB is not None:  # If an updated box is available
                self.initBB = self.updatedBB  # Get the updated box
                self.updatedBB = None  # Reset the updatedBB
                # Re-create the tracker
                self.tracker = self._initialize_tracker(self.tracker_name,
                                                        update=True)
                # Initialize the tracker
                self.tracker.init(self.frame, tuple(self.initBB))
                # Restart the fps
                self.fps = FPS().start()

            else:
                # Get the updated bounding box from tracker
                success, BB = self.tracker.update(self.frame)

                if success:  # If succeded in tracking
                        x, y, w, h = [int(item) for item in BB]
                        cv2.rectangle(self.frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)

                # Update the FPS counter
                self.fps.update()
                self.fps.stop()

                # Put some info on the frame
                info = [
                        ('Tracker', self.tracker_name),
                        ('Success', 'Yes' if success else 'No'),
                        ('FPS', f'{round(self.fps.fps(), 2)}')
                    ]

                for i, (label, value) in enumerate(info):
                    text = f"{label}: {value}"
                    cv2.putText(
                        self.frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

            cv2.imshow(self.window[0], self.frame)
            key = cv2.waitKey(1) & 0xFF  # Get the keycode for pressed key

            # If 'S' is pressed, re-initialize the bounding box
            if key == ord('s'):
                self._get_BB(update=True)  # Get a new bounding box
                self.fps = FPS().start()  # Restart the FPS counter

            elif (key == ord('q') or key == 27  # if 'Q' or 'ESC' is pressed,
                    or not cv2.getWindowProperty(  # or if the window is closed
                        self.window[0], cv2.WND_PROP_VISIBLE
                        )
                  ):

                print('Exiting...')
                break  # Stop the stream

            # Make sure that an update is not already in progress
            if not self.update_in_progress:
                self.frame_count += 1  # Increment the frame counter

            # Request a bounding box update if interval is reached.
            if self.frame_count == self.interval:
                self.update_in_progress = True
                self.frame_count = 0  # Reset the frame counter
                t = threading.Thread(target=self._get_BB, args=(True,))
                t.start()

        self.stop()  # Release the resources and cleanup

    def stop(self):
        """
        Tracker.stop(self)

        Releases resources. Destroys all OpenCV Windows and releases
        file pointer to the video (if one was given) or stops the webcam
        (otherwise).
        """
        # If we were using webcam, stop it
        if self.using_webcam:
            self.vs.stop()

        # Otherwise, release the file pointer to tje video provided
        else:
            self.vs.release()

        # Close all windows
        cv2.destroyAllWindows()


# For testing this module, simply run this script itself
if __name__ == '__main__':
    import sys

    # Please ensure the following files are present
    # ./deploy.prototxt.txt
    # ./res10_300x300_ssd_iter_140000.caffemodel

    # A test example
    try:
        tracker = Tracker(
            detector=("../deploy.prototxt.txt",
                      "../res10_300x300_ssd_iter_140000.caffemodel"),
            confidence=0.4,
            tracker='kcf',
            refresh_interval=5,
            video=None,
            width=400,
            window=None
          )

    except FileNotFoundError:  # If the files are not found
        print("The following files were not found in the current directory,"
              "exiting...\n"
              "\t1. deploy.prototxt.txt, "
              "\t2. res10_300x300_ssd_iter_140000.caffemodel")
        sys.exit(0)

    tracker.start()
