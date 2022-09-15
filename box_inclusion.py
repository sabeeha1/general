import cv2
import pyautogui


class BoundingBoxWidget(object):
    def __init__(self, img,scale_image):
        self.scale_image = scale_image
        self.original_image = img
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')

        self.clone = cv2.putText(self.clone, 'Draw a box around item. ', (10,40),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 .4, (255, 0, 0),1, cv2.LINE_AA)
        self.clone = cv2.putText(self.clone, 'Press right Click to undo. Press \'Q\' to save ', (10,60),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 .4, (255, 0, 0),1, cv2.LINE_AA)

        cv2.setMouseCallback('image', self.extract_coordinates)
        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        global box_x, box_y, box_width, box_height
        # Record starting (x,y) coordinates on left mouse button click
        if  event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.width = int((self.image_coordinates[1][0] - self.image_coordinates[0][0])/self.scale_image)
            self.height = int((self.image_coordinates[1][1] - self.image_coordinates[0][1]) / self.scale_image)
            # print('x,y,w,h : ({}, {}, {}, {})'.format(int(self.image_coordinates[0][0]/self.scale_image + self.width/2), \
            #                                           int(self.image_coordinates[0][1]/self.scale_image + self.height/2), \
            #                                           self.width,self.height))

            box_x, box_y, box_width, box_height = int(self.image_coordinates[0][0]/self.scale_image + self.width/2), \
                                                      int(self.image_coordinates[0][1]/self.scale_image + self.height/2), \
                                                      self.width,self.height
            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)
        if event == cv2.EVENT_RBUTTONDOWN:
           self.clone = self.original_image.copy()
           self.clone = cv2.putText(self.clone, 'Draw a box around item. ', (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .4, (255, 0, 0), 1, cv2.LINE_AA)
           self.clone = cv2.putText(self.clone, 'Press right Click to undo. Press \'Q\' to save ', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .4, (255, 0, 0), 1, cv2.LINE_AA)

    def show_image(self):
        return self.clone

def box_draw(file_name):

    width_screen, height_screen = pyautogui.size()
    img = cv2.imread(file_name)  # Read image
    # print(width_screen, height_screen)
    height_image, width_image, _ = img.shape
    print(width_image, height_image)
    scale_image = 1
    scale_width, scale_height = width_screen / width_image, height_screen / height_image
    if scale_width < 1 or scale_height < 1:
        if scale_width < 1:
            scale_image = scale_width
        elif scale_height < 1:
            scale_image = scale_height
        print(scale_image)
        img = cv2.resize(img, (int(width_image * scale_image), int(height_image * scale_image)))
    boundingbox_widget = BoundingBoxWidget(img,scale_image)
    while True:
        # cv2.moveWindow('image', 40, 30)
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)
        box = dict()
        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            box = {"name":"rect","x":box_x,"y":box_y,"width":box_width,"height":box_height}
            break
    return box

box_x,box_y,box_width,box_height =0,0,0,0
rec = box_draw("test.png")
print(rec)