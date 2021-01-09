# linux使用技巧
## 在thinkpad上调整trackpoint速度
使用`xinput`可以查看输入设备，命令为：
```
sudo xinput --list --short
```
从中找到小红点Trackpoint对应的设备号：
```
⎡ Virtual core pointer                    	id=2	[master pointer  (3)]
⎜   ↳ Virtual core XTEST pointer              	id=4	[slave  pointer  (2)]
⎜   ↳ SynPS/2 Synaptics TouchPad              	id=12	[slave  pointer  (2)]
⎜   ↳ TPPS/2 IBM TrackPoint                   	id=13	[slave  pointer  (2)]
⎣ Virtual core keyboard                   	id=3	[master keyboard (2)]
    ↳ Virtual core XTEST keyboard             	id=5	[slave  keyboard (3)]
    ↳ Power Button                            	id=6	[slave  keyboard (3)]
    ↳ Video Bus                               	id=7	[slave  keyboard (3)]
    ↳ Video Bus                               	id=8	[slave  keyboard (3)]
    ↳ Sleep Button                            	id=9	[slave  keyboard (3)]
    ↳ Integrated Camera: Integrated C         	id=10	[slave  keyboard (3)]
    ↳ AT Translated Set 2 keyboard            	id=11	[slave  keyboard (3)]
    ↳ ThinkPad Extra Buttons                  	id=14	[slave  keyboard (3)]
```
可以找到`id=13`的设备是trackpoint，然后修改移动速度：
```
sudo  xinput --set-prop "13" "Device Accel Constant Deceleration" 0.5
```
其中`set-prop`选择设备号，移动速度可选值为0-1000，数值越小移动速度越快。