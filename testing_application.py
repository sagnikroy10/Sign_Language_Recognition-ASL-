import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Initialize the webcam, detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
          "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


true_labels = []  # This will automatically populated with predicted labels
predicted_labels = []  # This will hold the predicted labels from the model

accuracy_display = 0  
frame_limit = 60  # Number of frames to collect for confusion matrix
frame_count = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100  
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100  

        if abs(accuracy_display - accuracy) > 0.5: 
            accuracy_display += (accuracy - accuracy_display) * 0.1 

        # Draw the predicted label and accuracy on the image
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        
        # Display label and accuracy in the format: Label: Accuracy%
        cv2.putText(imgOutput, f"{labels[index]}: {accuracy_display:.1f}%", 
                    (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, 
                    (255, 105, 180), 2) 
        
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Show the crop and the white background image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        true_labels.append(labels[index])  
        predicted_labels.append(labels[index])  

        frame_count += 1

        # If enough frames have been collected, calculate the confusion matrix
        if frame_count >= frame_limit:
            cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
            print("Confusion Matrix:")
            print(cm)

            # Plot confusion matrix using Matplotlib
            plt.figure(figsize=(10, 7))
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xticks(np.arange(len(labels)), labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

            # Display confusion matrix numbers inside the bounding boxes
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j] > 0:
                        cv2.putText(imgOutput, f'{cm[i, j]}', 
                                    (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (255, 255, 255), 1)  # Small white text

            break  

    cv2.imshow("Image", imgOutput)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()  
cv2.destroyAllWindows()  
