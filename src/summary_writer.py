"""SummaryWriter for logging log values to a CSV file."""

import os
import csv


class LocalSummaryWriter:
    """A simple SummaryWriter that writes values to a CSV file."""

    def __init__(self, log_dir):
        self._log_dir = log_dir
        self.file_path = os.path.join(log_dir, "history.txt")
        self.parameters_file_path = os.path.join(log_dir, "parameters.txt")
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        # Initialize the CSV file with headers if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["tag", "scalar_value", "global_step"])

    @property
    def log_dir(self):
        """Return the log directory."""
        return self._log_dir

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Append a scalar value to the CSV file."""
        # Append a new scalar value to the CSV file
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([tag, scalar_value, global_step if global_step is not None else ""])

    def add_parameters(self, parameters):
        """Save the parameters to a text file."""
        # Save the parameters to a text file
        with open(self.parameters_file_path, mode="w") as file:
            for key, value in parameters.items():
                file.write(f"{key}: {value}\n")

    def close(self):
        """Close the SummaryWriter."""
        # Placeholder for compatibility with TensorFlow's SummaryWriter
        pass
