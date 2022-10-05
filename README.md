# NerF Interface
ROS Wrapper for the NerF Simulator
## Overview
#### Description
The NerF Interface takes the ARL Simulator from arpl_quadrotor control and feeds the drone's pose into the NerF Simulator. Predicted Rendered Views from the NerF Simulator are then sent back to the Interface and are published on a ROS topic as images.

## ROS Organization
The ROS Organization is shown in the figure below. 
![Screenshot](doc/ros_diagram.png)
The table below also summarized the publication and subscription scheme.
|Name|Description|Publications|Subscriptions|
|---|---|---|---|
|`/nerf_interface`|Links ARL Simulator With NerF Simulator|`/renderedView`|`/quadrotor/pose`

## Dependencies and Installation
In order to use this package, first follow the instructions for the following two repositories and their respective dependencies:
- [arpl_quadrotor_control](https://github.com/arplaboratory/arpl_quadrotor_control).
- [nerf_simulation](https://github.com/arplaboratory/nerf_simulation).

After installation of necessary packages, clone the repo and `catkin_make` the ROS workspace. Source the `setup.bash` file inside the devel folder.

```
$ cd /path/to/your/workspace/src
$ git clone git@github.com:Zedai/nerf_interface.git
$ catkin_make
$ source ~/path/to/your/workspace/devel/setup.bash
```

##  Running
### Initialize Simulation
Directly call the `.launch`  file using `roslaunch` command:
```
$ source ~/path/to/your/workspace/devel/setup.bash
$ roslaunch nerf_interface nerf_sim.launch
$ python ~/path/to/your/workspace/src/nerf_interface/scripts/test.py
```
### Start simulation
Now you should have everything set up and properly linked together. Now you can control the drone using rqt_mav_manager and the rendered view should be streamed to your rviz display. Consult [arpl_quadrotor_control](https://github.com/arplaboratory/arpl_quadrotor_control) for more information on how to use rqt_mav_manager and the ARL Simulator.
