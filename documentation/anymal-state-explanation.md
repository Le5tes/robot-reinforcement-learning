# Anymal State Explanation

The observation returned from when we call the "step" function contains the following fields:
- t (time)
- state
- sensors 

## State
State contains two fields:
- Q
- V

### Q - position
This field contains the position of the body an 3D space and the positions of each of the joints in their own space (i.e. the joint angles)

This is given as an array, where the values are in this order:
- Body X position
- Body Y position
- Body Z position
- Body X Rotation/orientation
- Body Y Rotation/orientation
- Body Z Rotation/orientation
- Body W Rotation/orientation
- Left Front limb hip abductor/adductor joint angle
- Left Front limb hip flex/extend joint angle
- Left Front limb knee flex/extend joint angle
- Left Hind limb hip abductor/adductor joint angle
- Left Hind limb hip flex/extend joint angle
- Left Hind limb knee flex/extend joint angle
- Right Front limb hip abductor/adductor joint angle
- Right Front limb hip flex/extend joint angle
- Right Front limb knee flex/extend joint angle
- Right Hind limb hip abductor/adductor joint angle
- Right Hind limb hip flex/extend joint angle
- Right Hind limb knee flex/extend joint angle

N.B: Abductor/adductor means movement away/towards the midline, (like how your legs move doing star jumps!)
flex/extend is forwards/backwards movement (like kicking a football)

### V - velocity
This field contains the velocity of the body an 3D space and the velocities of each of the joints in their own space (i.e. the joint angles)

- Body X linear velocity
- Body Y linear velocity
- Body Z linear velocity
- Body X angular velocity
- Body Y angular velocity
- Body Z angular velocity
- Left Front limb hip abductor/adductor joint velocity
- Left Front limb hip flex/extend joint velocity
- Left Front limb knee flex/extend joint velocity
- Left Hind limb hip abductor/adductor joint velocity
- Left Hind limb hip flex/extend joint velocity
- Left Hind limb knee flex/extend joint velocity
- Right Front limb hip abductor/adductor joint velocity
- Right Front limb hip flex/extend joint velocity
- Right Front limb knee flex/extend joint velocity
- Right Hind limb hip abductor/adductor joint velocity
- Right Hind limb hip flex/extend joint velocity
- Right Hind limb knee flex/extend joint velocity

## Sensors
This returns arrays of the values of sensors. There are four types of sensors:
- ImuSensor (motion sensors- accelerometers, gyroscopes etc. )
- ForceSensor
- EncoderSensor (Joint positions sensors)
- EffortSensor

When using a real robot we wouldn't have the state, but would use these sensors instead to get an approximation.

I'm not sure if we want to use these so I haven't looked into them in detail! 
I think just using the state is sufficient for our purposes.
