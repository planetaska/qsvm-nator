#!/usr/bin/env python3

import os
import subprocess
import sys


class ExperimentFacade:
    """
    Facade class to provide a simple interface for running various experiments.
    """

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.experiments = {
            1: {
                "name": "WDBC",
                "path": os.path.join(self.base_dir, "exp_wdbc", "wdbc.py"),
                "description": "Wisconsin Diagnostic Breast Cancer dataset experiment"
            },
            2: {
                "name": "AdHoc QSVC",
                "path": os.path.join(self.base_dir, "exp_adhoc", "ah1_qsvc.py"),
                "description": "Ad Hoc experiment with Quantum Support Vector Classification"
            },
            3: {
                "name": "AdHoc Cluster",
                "path": os.path.join(self.base_dir, "exp_adhoc", "ah2_cluster.py"),
                "description": "Ad Hoc experiment with Clustering"
            },
            4: {
                "name": "AdHoc KPCA",
                "path": os.path.join(self.base_dir, "exp_adhoc", "ah3_kpca.py"),
                "description": "Ad Hoc experiment with Kernel Principal Component Analysis"
            }
        }

    def display_menu(self):
        """Display the experiment menu to the user."""
        print("\n" + "=" * 50)
        print("QSVM-nator Experiment Runner".center(50))
        print("=" * 50)

        for key, exp in self.experiments.items():
            print(f"{key}. {exp['name']} - {exp['description']}")

        print("q. Exit")
        print("-" * 50)

    def run_experiment(self, choice):
        """Run the selected experiment script."""
        if choice not in self.experiments:
            print(f"Invalid choice: {choice}")
            return False

        experiment = self.experiments[choice]
        print(f"\nRunning experiment: {experiment['name']}...")
        print(f"Executing script: {experiment['path']}\n")

        try:
            # Use sys.executable to ensure we use the same Python interpreter
            result = subprocess.run(
                [sys.executable, experiment['path']],
                check=True,
                text=True
            )
            print(f"\nExperiment {experiment['name']} completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nError running experiment: {e}")
            return False

    def run_with_args(self, exp_number, *args):
        """Run an experiment with command line arguments."""
        if exp_number not in self.experiments:
            print(f"Invalid experiment number: {exp_number}")
            return False

        experiment = self.experiments[exp_number]
        print(f"\nRunning experiment: {experiment['name']} with arguments: {' '.join(args)}")

        try:
            cmd = [sys.executable, experiment['path']] + list(args)
            result = subprocess.run(cmd, check=True, text=True)
            print(f"\nExperiment {experiment['name']} completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nError running experiment: {e}")
            return False

    def run_interactive(self):
        """Run the facade in interactive mode with a console UI."""
        while True:
            self.display_menu()
            try:
                choice = input("Enter your choice (1-4 or q to exit): ")
                if not choice.strip():
                    continue

                if choice.lower() == 'q':
                    print("Exiting program. Goodbye!")
                    break

                try:
                    choice = int(choice)
                except ValueError:
                    print("Please enter a number between 1-4 or 'q' to exit.")
                    continue

                if choice in self.experiments:
                    # Ask if user wants to provide arguments
                    args_input = input("Enter any command line arguments (or press Enter for none): ")
                    args = args_input.split() if args_input.strip() else []

                    if args:
                        self.run_with_args(choice, *args)
                    else:
                        self.run_experiment(choice)
                else:
                    print(f"Invalid choice: {choice}. Please select 1-4 or 'q' to exit.")

            except ValueError:
                print("Please enter a number between 1 and 4 or 'q' to exit.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user. Exiting...")
                break

            input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Check if specific experiment was requested via command line
    if len(sys.argv) > 1:
        try:
            exp_number = int(sys.argv[1])
            facade = ExperimentFacade()
            if exp_number in facade.experiments:
                # Pass any additional arguments to the experiment
                additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
                facade.run_with_args(exp_number, *additional_args)
            else:
                print(f"Invalid experiment number: {exp_number}")
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Please provide a number.")
    else:
        # Run in interactive mode if no arguments provided
        facade = ExperimentFacade()
        facade.run_interactive()