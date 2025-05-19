class PID:
    def __init__(self, proportion=0.5, integral=0, derivative=0.1, dt=0.1):
        self.proportion = proportion
        self.integral = integral
        self.derivative = derivative
        self.dt = dt
        self.prev_error = 0
        self.integral_sum = 0

    def calculate(self, error):
        self.integral_sum += error * self.dt
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        self.prev_error = error

        return (
            self.proportion * error +
            self.integral * self.integral_sum +
            self.derivative * derivative
        )
