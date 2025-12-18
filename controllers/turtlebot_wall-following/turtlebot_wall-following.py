"""turtlebot_wall-following controller"""

# You can us this template to implement the behaviors and maze sover state-machine 
# for the Turtlebot3. 

from controller import Robot #, DistanceSensor, Motor, Compass
import numpy as np

#-------------------------------------------------------
# Initialize variables

MAX_SPEED = 40 #6.28

# create the Robot instance.
robot = Robot()
pulses_per_turn = 4096    # number of pulses per wheel turn (encoder resolution)

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())   # [ms]
delta_t = robot.getBasicTimeStep()/1000.0    # [s]

# states
states = ['forward', 'decide_turn', 'turn_left', 'turn_right', 'turn_around', 'backward', 'stop']
current_state = states[0]

MAX_W = 1.5  # rad/s


# counter: used to maintain an active state for a number of cycles
counter = 0
COUNTER_MAX = 50  # minimum time before checking alignment
COUNTER_BEF_TURN = 3 # wait time before deciding turn

# Robot wheel speeds
wl = 0.0    # angular speed of the left wheel [rad/s]
wr = 0.0    # angular speed of the right wheel [rad/s]

# Robot linear and angular speeds
u = 0.0    # linear speed [m/s]
w = 0.0    # angular speed [rad/s]

# Physical parameters for the kinematics model (constants)
R = 0.330    # radius of the wheels [m]
D = 0.160    # distance between the wheels [m]

# Controller gains - define how the reaction of the robot will be:
# higher controller gain will result in faster reaction, but it can cause oscillations
k_1 = 1
k_2 = 1

#-------------------------------------------------------
# Initialize devices

# 2D-LiDAR
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)
lidar.enablePointCloud()
pointCloud = []    # list to store the point cloud data

# encoders
encoder = []
encoderNames = ['left wheel sensor', 'right wheel sensor']  # defined in the robot model
for i in range(2):
    encoder.append(robot.getDevice(encoderNames[i]))
    encoder[i].enable(timestep)
    
oldEncoderValues = []

# motors    
leftMotor = robot.getDevice('left wheel motor')        # name defined in the robot model
rightMotor = robot.getDevice('right wheel motor')      # name defined in the robot model
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)


#######################################################################
# Functions
# write here any function you may need: wheel speed computation, robot speed computation,
# wall-following behavior, etc.

def wheel_speed_commands(u_d, w_d, D, R):
    """Converts desired speeds to wheel speed commands
    Inputs:
        u_d = desired linear speed for the robot [m/s]
        w_d = desired angular speed for the robot [rad/s]
        R = radius of the robot wheel [m]
        D = distance between the left and right wheels [m]
    Returns:
        wr_d = desired speed for the right wheel [rad/s]
        wl_d = desired speed for the left wheel [rad/s]
    """
    wr_d = float((2*u_d + D*w_d)/(2*R))
    wl_d = float((2*u_d - D*w_d)/(2*R))

    return wl_d, wr_d

def get_wheels_speed(encoderValues, oldEncoderValues, pulses_per_turn, delta_t):
    """Computes speed of the wheels based on encoder readings
    """
    # Calculate the change in angular position of the wheels:
    ang_diff_l = 2*np.pi*(encoderValues[0] - oldEncoderValues[0])/pulses_per_turn
    ang_diff_r = 2*np.pi*(encoderValues[1] - oldEncoderValues[1])/pulses_per_turn

    # Calculate the angular speeds:
    wl = ang_diff_l/delta_t
    wr = ang_diff_r/delta_t

    return wl, wr

def get_robot_speeds(wl, wr, R, D):
    """Computes robot linear and angular speeds"""
    u = R/2.0 * (wr + wl)
    w = R/D * (wr - wl)
    
    return u, w

def avg_distance(pointCloud, center, angle_range=5):
    """Computes the average distance to an obstacle in a given sector of the LiDAR data"""
    
    start = center - angle_range // 2
    end = center + angle_range // 2
    
    distances = []
    for angle in range(start, end + 1):
        # garante que o índice fica entre 0-359
        idx = angle % 360
        distances.append(pointCloud[idx])
    
    result = np.mean(distances)

    # w_desired = np.clip(w_desired, -MAX_W, MAX_W)

    if result == float('inf'):
        return 0
    return result


def corridor_following_control(RIGHT_Distance, LEFT_Distance, u_desired = 1.5, k_angular = 1.5, dead_zone = 0.1):
    """Keeps robot centered in corridor by adjusting angular velocity"""

    lat_error = LEFT_Distance - RIGHT_Distance
    if (abs(lat_error) < dead_zone) or (abs(lat_error) > 0.75):
        return u_desired, 0.0
    
    if lat_error < 0:
        dead_zone = -dead_zone

    w_desired = k_angular * (lat_error - dead_zone)

    print(f' w_desired: {w_desired}, lat_error: {lat_error}')

    return u_desired, w_desired

def corridor_following_control_angle(LEFT_Distance, RIGHT_Distance, LEFT_Angle, RIGHT_Angle, u_desired = 2.0, k_linear = 2.5, k_angular = 20.0, dead_zone = 0.1):
        # Wall-following on both sides
        lat_error = LEFT_Distance - RIGHT_Distance
        angular_error = RIGHT_Angle - LEFT_Angle
        
        if abs(angular_error) > 0.4: angular_error = 0

        if abs(lat_error) < 0.125:
            w_desired = k_angular * angular_error * 25.0  # 3x more aggressive on angle!
        else:

            if (abs(lat_error) < dead_zone) or (abs(lat_error) > 0.7):
                lat_error = 0

            w_desired = k_linear * lat_error + k_angular * angular_error

        w_desired = np.clip(w_desired, -MAX_W, MAX_W)

        print(f'Corridor following: LEFT_Distance = {LEFT_Distance:.2f} m, RIGHT_Distance = {RIGHT_Distance:.2f} m, LEFT_Angle = {LEFT_Angle:.2f} rad, RIGHT_Angle = {RIGHT_Angle:.2f} rad, w_d = {w_desired:.2f} rad/s, lat_error = {lat_error:.2f} m, angular_error = {angular_error:.2f} rad.')

        return u_desired, w_desired

def distance_to_point(distance, sensor_angle):
    angle_rad = np.radians(sensor_angle)
    if distance == float('inf'):
        distance = 10.0 
    x = distance * np.cos(angle_rad) 
    y = distance * np.sin(angle_rad)
    return x, y

def relative_angle(x1, y1, x2, y2):
    angle = np.arctan2(y2 - y1, x2 - x1)
    return angle

#######################################################################
# Main loop: See-think-act cycle
# Perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    #----------------------------- See -----------------------------------
    # Read the LiDAR data
    pointCloud = lidar.getRangeImage()  # [m], 360 values, one per degree

    # See every directions
    BACK_Distance = avg_distance(pointCloud, 0)     # South
    RIGHT_Distance = avg_distance(pointCloud, 270)    # East
    FRONT_Distance = avg_distance(pointCloud, 180, angle_range=20)   # North
    LEFT_Distance = avg_distance(pointCloud, 90)   # West
    # print(f'Distances: FRONT={FRONT_Distance:.2f} m, RIGHT={RIGHT_Distance:.2f} m, BACK={BACK_Distance:.2f} m, LEFT={LEFT_Distance:.2f} m')

    x_l, y_l = distance_to_point(avg_distance(pointCloud, 95, 1), 95)
    x_l_2, y_l_2 = distance_to_point(avg_distance(pointCloud, 85, 1), 85)
    LEFT_Angle = relative_angle(x_l, y_l, x_l_2, y_l_2)

    x_r, y_r = distance_to_point(avg_distance(pointCloud, 265, 1), 265)
    x_r_2, y_r_2 = distance_to_point(avg_distance(pointCloud, 275, 1), 275)
    RIGHT_Angle = -relative_angle(x_r, y_r, x_r_2, y_r_2)
       
    encoderValues = []
    for i in range(2):
        encoderValues.append(encoder[i].getValue())    # [rad]
        
    # Update old encoder values if not done before
    if len(oldEncoderValues) < 2:
        for i in range(2):
            oldEncoderValues.append(encoder[i].getValue())   

    # Compute wheels speeds
    wl, wr = get_wheels_speed(encoderValues, oldEncoderValues, pulses_per_turn, delta_t)
    # print(f'Left wheel speed  = {wl} rad/s.')
    # print(f'Right wheel speed = {wr} rad/s.')

    # Compute robot speeds
    u, w = get_robot_speeds(wl, wr, R, D)
    # print(f"Robot linear speed  = {u} m/s")
    # print(f"Robot angular speed = {w} rad/s")

    #----------------------------- Think ---------------------------------
    # Implement the finite-state machine to select the robot behavior

    # ========== BEHAVIORS: What to do in each state ==========
    if current_state == 'forward':
        # u_d , w_d = corridor_following_control(RIGHT_Distance, LEFT_Distance)
        u_d , w_d = corridor_following_control_angle(LEFT_Distance, RIGHT_Distance, LEFT_Angle, RIGHT_Angle)
        # print(f'Corridor following control: u_d = {u_d:.2f}, w_d = {w_d:.2f}')

    if current_state == 'decide_turn':
        u_d = 0.0  # stop to decide
        w_d = 0.0   

    if current_state == 'turn_left':
        u_d = 0.45
        w_d = 10.0  # turn left

    if current_state == 'turn_right':
        u_d = 0.45
        w_d = -10.0  # turn right

    if current_state == 'turn_around':
        u_d = 0.0
        w_d = 18.0  # turn 180°

    if current_state == 'backward':
        u_d = -0.5
        w_d = 0.0


    # ========== TRANSITIONS: When to change state ==========
    if current_state == 'forward':
        if FRONT_Distance < 0.25:
            if counter >= COUNTER_BEF_TURN:
                current_state = 'decide_turn'
                counter = 0

        # if RIGHT_Distance > 2.5:
        #     current_state = 'turn_right'
        #     counter = 0

    if current_state == 'decide_turn':
        THRESHOLD_OPEN = 0.3  # minimum clear distance
        
        left_open = LEFT_Distance > THRESHOLD_OPEN
        right_open = RIGHT_Distance > THRESHOLD_OPEN
        front_open = FRONT_Distance > THRESHOLD_OPEN
        
        print(f'  >> DECISION: L={LEFT_Distance:.2f}[{left_open}] R={RIGHT_Distance:.2f}[{right_open}] F={FRONT_Distance:.2f}[{front_open}]')
        
        # IMPROVED DECISION LOGIC
        if front_open and FRONT_Distance > 0.4:
            # False alarm, front is clear
            current_state = 'forward'
            print('  >> Front clear - CONTINUE FORWARD')
        # elif not left_open and not right_open and not front_open:
        #     # TRUE DEAD END - turn around
        #     current_state = 'turn_around'
        #     print('  >> DEAD END - TURN AROUND')
        elif left_open and right_open:
            # Both open - choose better path
            if RIGHT_Distance > LEFT_Distance:
                current_state = 'turn_right'
                print(f'  >> Both open - RIGHT better ({RIGHT_Distance:.2f} > {LEFT_Distance:.2f})')
            elif LEFT_Distance > RIGHT_Distance:
                current_state = 'turn_left'
                print(f'  >> Both open - LEFT better ({LEFT_Distance:.2f} > {RIGHT_Distance:.2f})')
            else:
                # Similar - use right-hand rule
                current_state = 'turn_right'
                print('  >> Both open similar - RIGHT (right-hand rule)')
        elif right_open:
            current_state = 'turn_right'
            print('  >> Only RIGHT open')
        elif left_open:
            current_state = 'turn_left'
            print('  >> Only LEFT open')
        else:
            # Shouldn't reach here, but turn around just in case
            current_state = 'turn_around'
            print('  >> Fallback - TURN AROUND')
        
        counter = 0
            

    if current_state == 'turn_left':
        if counter >= COUNTER_MAX:  # completed turn
            current_state = 'forward'
            counter = 0

    if current_state == 'turn_right':
        if counter >= COUNTER_MAX:  # completed turn
            current_state = 'forward'
            counter = 0

    if current_state == 'turn_around':
        if counter >= COUNTER_MAX:  # 180° takes longer
            current_state = 'forward'
            counter = 0

    # increment counter
    counter += 1

    # update old encoder values for the next cycle
    oldEncoderValues = encoderValues
    
    #----------------------------- Act -----------------------------------
    # Set motor speeds with the values defined by the state-machine
    leftSpeed, rightSpeed = wheel_speed_commands(u_d, w_d, D, R)
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

    # Debug
    # print(f'Current state = {current_state}, distances = {pointCloud[180]:.2f}, u_d = {u_d:.2f}, w_d = {w_d:.2f}')
    
    print(f'Current state = {current_state}, Distances: FRONT={FRONT_Distance:.2f} m, RIGHT={RIGHT_Distance:.2f} m, Back={BACK_Distance:.2f} m, LEFT={LEFT_Distance:.2f} m')
    # Repeat all steps while the simulation is running.

