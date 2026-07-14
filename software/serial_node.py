import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Imu
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import serial
import struct
import math
import threading
import sys
import time



class STM32Node(Node):
    def __init__(self):
        super().__init__('stm32_node')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 9600)

        self.port_name = self.get_parameter('port').get_parameter_value().string_value
        self.baud_rate = self.get_parameter('baudrate').get_parameter_value().integer_value
        

        try:
            self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=0.01)
            self.get_logger().info(f"Connect!: {self.port_name} @ {self.baud_rate}")
        except Exception as e:
            self.get_logger().error(f"Error ({self.port_name}): {e}")
            return
        

        self.data_buffer = bytearray()       
        self.text_accumulator = bytearray()  
        self.last_buttons = [0] * 12
        self.last_axes = [0.0] * 8
        self.monitoring_active = False


        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.rpy_pub = self.create_publisher(Float32MultiArray, '/imu/rpy_deg', 10)
        self.tf_broadcaster = TransformBroadcaster(self) 


        self.create_timer(0.01, self.read_serial_daemon)


        input_thread = threading.Thread(target=self.console_input_loop, daemon=True)
        input_thread.start()

        self.get_logger().info("Ready!")

    # =========================================
    # =========================================
    def console_input_loop(self):
        while True:
            try:
                cmd = input() 
                if cmd.strip():
                    full_cmd = cmd + "\n"
                    if self.ser and self.ser.is_open:
                        self.ser.write(full_cmd.encode('utf-8'))
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"Input Error: {e}")

    # =========================================
    # =========================================
    def read_serial_daemon(self):
        if not self.ser or not self.ser.is_open: return

        try:
            waiting = self.ser.in_waiting
            if waiting > 0:
                new_data = self.ser.read(waiting)
                self.data_buffer.extend(new_data)
                self.parse_mixed_buffer()
        except Exception as e:
            self.get_logger().error(f"Read Error: {e}")

    def parse_mixed_buffer(self):
        while len(self.data_buffer) > 0:
            # IMU (0x55 0xAA)
            if len(self.data_buffer) >= 2 and self.data_buffer[0] == 0x55 and self.data_buffer[1] == 0xAA:
                if len(self.data_buffer) >= 11:
                    packet = self.data_buffer[0:11]
                    self.decode_imu(packet)
                    self.data_buffer = self.data_buffer[11:]
                    continue
                else:
                    break 

        
            if self.data_buffer[0] == 0x55 and len(self.data_buffer) == 1:
                break 

           
            byte_val = self.data_buffer.pop(0)
            if byte_val == 10:
                try:
                    line = self.text_accumulator.decode('utf-8', errors='ignore').strip()
                    if line: self.get_logger().info(f"[STM32]: {line}")
                except: pass
                self.text_accumulator.clear()
            elif byte_val != 13: 
                self.text_accumulator.append(byte_val)

    def decode_imu(self, packet):
        # 55 AA [Roll 4B] [Pitch 4B] [Sum 1B]
        payload = packet[2:10]
        checksum = packet[10]
        if (sum(payload) & 0xFF) == checksum:
            try:
                pitch, roll = struct.unpack('<ff', payload)
                self.publish_ros_msg(roll, pitch)
            except: pass

    # =========================================
    # =========================================
    def publish_ros_msg(self, roll, pitch):
        msg_array = Float32MultiArray()
        msg_array.data = [roll, pitch]
        self.rpy_pub.publish(msg_array)


        q = self.euler_to_quaternion(math.radians(roll), math.radians(pitch), 0)
        now = self.get_clock().now().to_msg()


        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)


        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"
        imu.orientation = t.transform.rotation
        self.imu_pub.publish(imu)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cx = math.cos(roll/2); sx = math.sin(roll/2)
        cy = math.cos(pitch/2); sy = math.sin(pitch/2)
        cz = math.cos(yaw/2); sz = math.sin(yaw/2)
        return [sx*cy*cz - cx*sy*sz, cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz, cx*cy*cz + sx*sy*sz]

    # =========================================
    # =========================================
    def joy_callback(self, msg):
        def is_pressed(idx): return msg.buttons[idx] == 1 and self.last_buttons[idx] == 0
        def is_holding(idx, dir): 
            if idx >= len(msg.axes): return False
            return msg.axes[idx] > 0.5 if dir > 0 else msg.axes[idx] < -0.5
        def was_holding(idx, dir):
            if idx >= len(self.last_axes): return False
            return self.last_axes[idx] > 0.5 if dir > 0 else self.last_axes[idx] < -0.5


        def is_axis_triggered(index, direction):
            # direction: 1.0 or -1.0
            if index >= len(msg.axes): return False
            val = msg.axes[index]
            last_val = self.last_axes[index]

            threshold = 0.5
            if direction > 0:
                return val > threshold and last_val <= threshold
            else:
                return val < -threshold and last_val >= -threshold
            

        # if is_holding(1, 1.0) and is_holding(4, 1.0) and not (was_holding(1, 1.0) and was_holding(4, 1.0)):
        #     self.send_cmd('m')
        

        if is_holding(1, 1.0) and is_holding(4, 1.0) and not (was_holding(1, 1.0) and was_holding(4, 1.0)):
            self.send_cmd('m')
            self.get_logger().info("TASK")


        # if is_holding(1, -1.0) and is_holding(4, -1.0) and not (was_holding(1, -1.0) and was_holding(4, -1.0)):
        #     self.send_cmd('n')

        

        if is_pressed(0): # A 
            self.send_cmd('i', "High pitch (i)")
        if is_pressed(1): # B 
            self.send_cmd('l', "right row (l)")
        if is_pressed(2): # X 
            self.send_cmd('j', "Left row (j)")
        if is_pressed(3): # Y 
            self.send_cmd('k', "Low pitch (k)")
        if is_pressed(4): # LB
            self.send_cmd('a', "Left Wing Open")
        if is_pressed(5): # RB
            self.send_cmd('d', "Right Wing Open")
        if is_pressed(6): # Select
            self.send_cmd('f', "Left Wing Bigger AOA")
        if is_pressed(7): # Start
            self.send_cmd('g',"Right Wing Bigger AOA")
        if is_pressed(8): # head logo
            self.send_cmd('p',"Print Message")
        if is_pressed(9): # L-Stick
            self.send_cmd('h',"Drange Water")
        if is_pressed(10): # R-Stick
            self.send_cmd('y',"Drink Water")
        if is_axis_triggered(6, 1.0): # 
            self.send_cmd('v', "Left Wing Smaller AOA")
        if is_axis_triggered(6, -1.0): # 
            self.send_cmd('b', "Right Wing Smaller AOA")
        if is_axis_triggered(7, 1.0): # 
            self.send_cmd('s', "Speed Up")
        if is_axis_triggered(7, -1.0): # 
            self.send_cmd('x', "Slow Down")    
        if is_axis_triggered(2, -1.0): # LT
            self.send_cmd('z', "Left Wing Close")  
        if is_axis_triggered(5, -1.0): # RT
            self.send_cmd('c', "Right Wing Close")  
        
        self.last_buttons = list(msg.buttons)
        self.last_axes = list(msg.axes)

    # def send_cmd(self, char_cmd):
    #     if self.ser and self.ser.is_open:
    #         self.ser.write(char_cmd.encode('utf-8'))
    # =========================================
    # =========================================
    def send_cmd(self, char_cmd, log_msg=""):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(char_cmd.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
                
def main(args=None):
    rclpy.init(args=args)
    node = STM32Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open: node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()