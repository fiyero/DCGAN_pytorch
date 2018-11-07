# DCGAN_pytorch<br/>
updatev1<br/>
(7th Nov 2018): when I write this model I didnt realise we no longer have to wrap tensor into variable in the latest pytorh<br/>
folder load_weight_results contains several fake generated img in different epoch

Run DCGAN in pytorch with multiple GPU for Cifar 10 dataset<br/>
The full code is for demonstration purpose, I have tried tunning different parameters for more than 100+ epochs.<br/>
Since it is a bit messy I just post the <b>original training condition </b>and it should be good enough to experience what is DCGAN.<br/>

fake img <b>before training</b><br/>
![fake_img_version_epoch_001](https://user-images.githubusercontent.com/38428076/48109499-b0860880-e282-11e8-8ca1-bed705e0eb0f.jpg)

fake img <b>after training</b><br/>
![fake_img_version_epoch_035](https://user-images.githubusercontent.com/38428076/48109457-7fa5d380-e282-11e8-8df5-696a6619cb66.jpg)

random real img<br/>
![real_img032](https://user-images.githubusercontent.com/38428076/48109580-fb078500-e282-11e8-83ab-1316427c65c2.jpg)<br/>

![image](https://user-images.githubusercontent.com/38428076/48110098-2f7c4080-e285-11e8-9fe4-65be34670139.png)<br/>


My problems are:<br/>
1.The quality of the fake img still have much room for improvement.<br/>
2.Sometime i will have weird result , such as<br/>
![image](https://user-images.githubusercontent.com/38428076/48109797-d6f87380-e283-11e8-849b-2c6ff216cdda.png)<br/>
You can see the generated fake imgs in the last few results are completely go wrong, but if training continues, it will kind of back to the inital untrained stage and train from the beginning again. <br/>
I guess this may due to large lr step?<br/>
3.The images of Cifar 10 dataset are low in resolution, I should try with larger img set to better experience DCGAN

