# Sefie_Drone
The Project idea is using Mediapipe pose and facial recognize (detech the controller face) to make the Drone auto follow the controller and take selfie, by gestrue control.  
I not really have the drone yet, this program using computer webcam instead for the drone camera to test the program.

## Turn on the program
There is two mode to turn on the program  
**Mode 1:** Directly turn on mediapipe pose.

![Mode_1_75](https://user-images.githubusercontent.com/81819640/124456774-4a483f80-ddb5-11eb-960c-de1d66a95e0a.png)  
The mediapipe Auto recognize the first person it's see, so i make the Mode 2, when it trun on the Drone will looking for the specific controller, come approach then turn on mediapipe.

**Mode 2:** Turn on facial recognize.
![Mode_2_75](https://user-images.githubusercontent.com/81819640/124458514-43223100-ddb7-11eb-8ab3-8edf438fbc6b.png)
## Controlling table
__Note:__ The Drone auto keep latitude and angle pointing to the controller, the program only detech pose to control it move lef, right, forward, backward, take picture, auto follow, and landing.
Pose | Action
---- | ----
**Cover/uncover camera three times** | **Turn on mode 1**
**Cover/uncover camera four times** | **Turn on mode 2**
![picture_1](https://user-images.githubusercontent.com/81819640/125103482-96251c80-e106-11eb-89bd-3dd363ca6c35.jpg) | **Pose: Right arm open <br/> Action: Go backward**
![picture_2](https://user-images.githubusercontent.com/81819640/125105115-4e06f980-e108-11eb-8602-423c49337098.jpg) | **Pose: Right arm close <br/> Action: Go forward**
![picture_3](https://user-images.githubusercontent.com/81819640/125105210-6d9e2200-e108-11eb-9ba6-06ff1650750c.jpg) | **Pose: Left arm open <br/> Action: Move right**
![picture_4](https://user-images.githubusercontent.com/81819640/125105298-84447900-e108-11eb-9bfc-ac229bfb2ae0.jpg) | **Pose: Left arm close <br/> Action: Move left**
![picture_5](https://user-images.githubusercontent.com/81819640/125105656-e4d3b600-e108-11eb-9e07-afaeee797643.jpg) | **Pose: Two hands cross <br/> Action: Take picture**
![picture_6](https://user-images.githubusercontent.com/81819640/125105736-fa48e000-e108-11eb-9aeb-0c34f50ddcf3.jpg) | **Pose: Two hands up <br/> Action: Keep/release keep** distance mode
![picture_7](https://user-images.githubusercontent.com/81819640/125105747-00d75780-e109-11eb-85ca-d9bbc8ae75a1.jpg) | Pose: left hands forward <br/> Action: Approach and land 
 
## Libraries and packages
### MediaPipe Pose:
I looking for some kind of light Pose Estimation with hight accuracy and running without GPU. The mediapipe pose is satisfy the condition and it's can achieves real-time performance on mobile phones for the future upgrade.
### Scikit-learn:
I Pick KNN Classification using Scikit-learn (KNN model) beacasue it's good at predict the controller and the unknow person (which mean not confused unknown person is the controller), KNN model don't need to fit mode again when you replace the controller data
### Facenet Architecture:
FaceNet model is 22-layers deep neural network that directly trains its output to be a 128-dimensional embedding (this is the input for the KNN-Model)
