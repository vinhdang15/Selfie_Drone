# Sefie_Drone
The Project idea is using Mediapipe pose and facial recognize (detech the controller face) to make the Drone auto follow the controller and take selfie, by gestrue control.  
I not really have the drone yet, this program using computer webcam instead the drone camera to test the program.

## Turn on the program
Mode 1: Directly turn on mediapipe pose.

![Mode_1_75](https://user-images.githubusercontent.com/81819640/124456774-4a483f80-ddb5-11eb-960c-de1d66a95e0a.png)  
The mediapipe Auto recognize the first person it's see, so i make the Mode 2, when it trun on the Drone will looking for the specific controller, come approach then turn on mediapipe.

Mode 2: Turn on facial recognize.
![Mode_2_75](https://user-images.githubusercontent.com/81819640/124458514-43223100-ddb7-11eb-8ab3-8edf438fbc6b.png)
## Pose table
Pose | Action
---- | ----
Cover/uncover camera three times | Turn on mode 1
Cover/uncover camera four times | Turn on mode 2
Pose: right arm open | Action: go backward
Pose: right arm close | Action: go forward
Pose: left arm close | Action: move left
Pose: left arm open | Action: move right
Pose: two hands up | Action: keep/release keep distance mode
Pose: two hands cross | Action: take picture
Pose: left hands forward | Action: Approach and land

## Libraries and packages
### MediaPipe Pose:
I looking for some kind of light Pose Estimation with hight accuracy and running without GPU. The mediapipe pose is satisfy the condition and it's can achieves real-time performance on mobile phones for the future upgrade.
### Scikit-learn:
I Pick KNN Classification using Scikit-learn (KNN model) beacasue it's good at predict the controller and the unknow person (which mean not confused unknown person is the controller), KNN model don't need to fit mode again when you replace the controller data
### Facenet Architecture:
FaceNet model is 22-layers deep neural network that directly trains its output to be a 128-dimensional embedding (this is the input for the KNN-Model)
