# DCGAN_pytorch

Run DCGAN in pytorch with multiple GPU for Cifar 10 dataset
The full code is for demonstration purpose, I have tried tunning different parameters for more than 100+ epochs.
Since it is a bit messy I just post the original training condition and it should be good enough to experience what is DCGAN.


fake img before training
![fake_img_version_epoch_001](https://user-images.githubusercontent.com/38428076/48109499-b0860880-e282-11e8-8ca1-bed705e0eb0f.jpg)

fake img after training
![fake_img_version_epoch_035](https://user-images.githubusercontent.com/38428076/48109457-7fa5d380-e282-11e8-8df5-696a6619cb66.jpg)

random real img
![real_img032](https://user-images.githubusercontent.com/38428076/48109580-fb078500-e282-11e8-83ab-1316427c65c2.jpg)

My problems are:<br/>
1.The quality of the fake img still have much room for improvement.<br/>
2.Sometime i will have weird result , such as<br/>
![image](https://user-images.githubusercontent.com/38428076/48109797-d6f87380-e283-11e8-849b-2c6ff216cdda.png)<br/>
You can see the generated fake imgs in the last few results are completely go wrong, but if training continues, it will kind of back to the inital untrained stage and train from the beginning again. I guess this may due to large lr step?
3.The images of Cifar 10 dataset are low in resolution, I should try with larger img set to better experience DCGAN
