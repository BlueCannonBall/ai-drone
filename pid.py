class PID:
	def __init__(self, proportion=0.5, integral=0.5, derivative=0.1, dt=0.1):
		self.proportion = proportion
		self.integral = integral
		self.derivative = derivative
		self.dt = dt
		self.prev_error = 0

	def calculate(error):
		integral = error * dt
		derivative = (error - self.prev_error) / dt if dt > 0 else 0
		self.prev_error = error
		return self.proportion * error + self.integral * integral + self.derivative * derivative
