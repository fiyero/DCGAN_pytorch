Run DCGAN in pytorch with multiple GPU for celebA dataset
The full code is for demonstration purpose, I have tried tunning different parameters.
Since it is a bit messy I just post the original training condition and it should be good enough to experience what is DCGAN.

fake img <b>before training</b><br/>
![fake_img_version_epoch_001](https://user-images.githubusercontent.com/38428076/48122037-84ce4700-e2b1-11e8-8bc3-6bc5d4cd5124.jpg)<br/>

fake img <b>after training</b><br/>
![fake_img_version_epoch_005](https://user-images.githubusercontent.com/38428076/48122054-8f88dc00-e2b1-11e8-94f6-fb9ecaa15385.jpg)<br/>

random real img<br/>
![image](https://user-images.githubusercontent.com/38428076/48122112-a4656f80-e2b1-11e8-916a-fdc4af0827ec.png)<br/>


![image](https://user-images.githubusercontent.com/38428076/48122139-b9420300-e2b1-11e8-8652-441afe0959f2.png)<br/>

Problems:<br/>
1. Hard to lower the LossG, still doing experiment on how to get the optimal training condition to lower LossG
