def remove_trailing_commas(filename):
    """Removes two trailing commas from each line in a file."""

    with open(filename, 'r') as file:
        lines = file.readlines()

    with open(filename, 'w') as file:
        for line in lines:
            if line.endswith(',,\n'):
                line = line[:-3] + '\n'  # Remove the last two commas
            file.write(line)

if __name__ == '__main__':
    filename = './sqliv3.csv'  # Replace with the actual filename
    remove_trailing_commas(filename)