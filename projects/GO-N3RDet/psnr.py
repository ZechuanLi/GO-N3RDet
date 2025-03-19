import re

def calculate_average_psnr_rmse_with_print(file_path):
    # Updated regex pattern to capture PSNR and RMSE values including scientific notation
    pattern = r'Sample 0: PSNR: ([\d.]+),.*?RMSE: \[\[\[([\d.eE+-]+)\s?\]'
    
    psnr_values = []
    rmse_values = []
    
    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Find matches for PSNR and RMSE
            match = re.search(pattern, line)
            if match:
                psnr_value = float(match.group(1))
                rmse_value = float(match.group(2))
                psnr_values.append(psnr_value)
                rmse_values.append(rmse_value)
                # Print extracted values
                print(f"Extracted PSNR: {psnr_value}, Extracted RMSE: {rmse_value}")
    
    # Calculate average PSNR and RMSE
    average_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    average_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else 0

    # Return the average values and total count
    return average_psnr, average_rmse, len(psnr_values)

# Example usage
file_path = '/home/zechuanl/mmdetection3d/myjob_output_471172.txt'  # Replace this with the path to your file
average_psnr, average_rmse, total_count = calculate_average_psnr_rmse_with_print(file_path)

print(f"\nAverage PSNR: {average_psnr}")
print(f"Average RMSE: {average_rmse}")
print(f"Total count: {total_count}")
