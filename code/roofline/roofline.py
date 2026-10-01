import sys
import csv

if len(sys.argv) <= 1:
    sys.exit('No profile output provided.')

kernel_parameters = {}

for profout_filename in sys.argv[1:]:

    # Clean/filter out lines that aren't part of the CSV table
    csv_lines = []
    with open(profout_filename, "r") as profout_file:
        skip_line = True
        
        for line in profout_file:
            if line.startswith('"Process ID"'):
                skip_line = False
        
            if not skip_line:
                csv_lines.append(line)

    # Parse CSV data and group parameters by kernel name
    reader = csv.DictReader(csv_lines)
    for row in reader:
        kernel_name = row["Kernel Name"]
        metric_name = row["Metric Name"]
        metric_val  = float(row["Average"])

        if kernel_name not in kernel_parameters:
            kernel_parameters[kernel_name] = {}
        
        kernel_parameters[kernel_name].update( {metric_name: metric_val} )

"""
# Print read parameters
for kernel, params in kernel_parameters.items():
  print(f"Kernel: {kernel}")
  print(f"Parameters: {params}\n")
"""

# Compute FLOPs, Bytes, and Time...
kernel_FBT = {}
for kernel, params in kernel_parameters.items():
    FLOPs = 0.0
    Bytes = 0.0
    Time  = params['sm__cycles_elapsed.avg'] / params['sm__cycles_elapsed.avg.per_second']

    for param_name, param_val in params.items():
        if 'sass_thread_inst_executed' in param_name:
            if 'dfma' in param_name or 'ffma' in param_name or 'hfma' in param_name:
                FLOPs += param_val * 2
            else:
                FLOPs += param_val
        
        elif 'bytes' in param_name:
            Bytes += param_val
    
    kernel_FBT[kernel] = {'FLOPs': FLOPs, 'Bytes': Bytes, 'Time': Time}


# Compute OI and GFLOPs
kernel_OI_GFLOPs = {}
for kernel, params in kernel_FBT.items():
    OI = params['FLOPs'] / params['Bytes']
    GFLOPs = (params['FLOPs'] / params['Time']) / 10e9
    
    kernel_OI_GFLOPs[kernel] = {'OI': OI, 'GFLOPs': GFLOPs}

# Print results
for kernel, params in kernel_OI_GFLOPs.items():
  print(f"\nKernel: {kernel}")
  print("Parameters:")
  for param_name, param_val in params.items():
    print(f"\t{param_name} = {param_val}")
print()