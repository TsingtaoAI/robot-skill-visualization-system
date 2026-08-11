
import pygame


class Joystick:
    def __init__(self, joystick_type: str):
        """Setup joystick for providing commands

        Args:
            joystick_type (str): type of joystick: xbox, switch
        """
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("No joystick connected")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print("Joystick name: {}".format(self.joystick.get_name()))
        if joystick_type not in ['xbox', 'switch']:
            raise ValueError("Unsupported joystick type: {}".format(joystick_type))
        if joystick_type == 'xbox':
            self.axis_id = {
                'LX': 0,
                'LY': 1,
                'RX': 3,
                'RY': 4,
                'LT': 2,
                'RT': 5
            }
            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }
        elif joystick_type == 'switch':
            self.axis_id = {
                'LX': 0,
                'LY': 1,
                'RX': 2,
                'RY': 3,
                'LT': 5,
                'RT': 4
            }
            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
            
    def update(self):
        """Update joystick state
        
        Axes:
            LX: Left stick X-axis
            LY: Left stick Y-axis
            RX: Right stick X-axis
            RY: Right stick Y-axis
        Buttons:
            X, Y, B, A: Face buttons
            LB, RB: Left and Right bumpers
            LT, RT: Left and Right triggers
        """
        pygame.event.pump()
        self.lx = self.joystick.get_axis(self.axis_id['LX'])
        self.ly = self.joystick.get_axis(self.axis_id['LY'])
        self.rx = self.joystick.get_axis(self.axis_id['RX'])
        self.ry = self.joystick.get_axis(self.axis_id['RY'])
        
        self.lt = self.joystick.get_axis(self.axis_id['LT']) > 0
        self.rt = self.joystick.get_axis(self.axis_id['RT']) > 0
        self.x = self.joystick.get_button(self.button_id['X'])
        self.y = self.joystick.get_button(self.button_id['Y'])
        self.b = self.joystick.get_button(self.button_id['B'])
        self.a = self.joystick.get_button(self.button_id['A'])
        self.lb = self.joystick.get_button(self.button_id['LB'])
        self.rb = self.joystick.get_button(self.button_id['RB'])
        self.select = self.joystick.get_button(self.button_id['SELECT'])
        self.start = self.joystick.get_button(self.button_id['START'])