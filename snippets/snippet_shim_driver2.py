# snippet_shim_driver.py - Game Controller Abstraction
# written by DeepSeek Chat (honor call: The Input Alchemist)

class UnifiedGameController:
    """Normalizes Xbox/PS/Switch controllers to common interface"""
    def __init__(self):
        self.controller = self._autodetect_controller()
        self.button_map = self._get_button_map()
        self.logger = logging.getLogger("GGFAI.controller")

    def _autodetect_controller(self) -> Any:
        try:
            import inputs
            devices = inputs.devices.gamepads
            if devices:
                return devices[0]
            raise ImportError("No gamepad found")
        except ImportError:
            self.logger.warning("Falling back to pygame controller")
            import pygame
            pygame.init()
            if pygame.joystick.get_count() > 0:
                return pygame.joystick.Joystick(0)
            return None

    def _get_button_map(self) -> Dict[str, int]:
        """Map physical buttons to unified names"""
        if isinstance(self.controller, inputs.Device):
            return {
                'confirm': inputs.BTN_A,
                'cancel': inputs.BTN_B,
                'menu': inputs.BTN_START
            }
        else:  # Pygame
            return {
                'confirm': 0,
                'cancel': 1,
                'menu': 9
            }

    def get_inputs(self) -> Dict[str, float]:
        """Returns normalized inputs (-1 to 1)"""
        if not self.controller:
            return {}
            
        if hasattr(self.controller, 'read'):  # inputs library
            events = self.controller.read()
            return {
                'left_x': next(
                    (e.state / 32768 for e in events if e.code == 'ABS_X'),
                    0.0
                )
            }
        else:  # Pygame
            return {
                'left_x': self.controller.get_axis(0)
            }