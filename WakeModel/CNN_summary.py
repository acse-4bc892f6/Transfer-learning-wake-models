from CNN_model import Generator
from torchsummary import summary

def GetModelSummary(write_to_file=True, file_name='cnn_summary.txt'):
    """
    Print CNN summary and save output to text file.
    
    Args:
        write_to_file (bool, optional): Save output to text file if True.
            Defaults to True.
        file_name (str, optional): File name of text file.
            Defaults to 'cnn_summary.txt'.
    """
    model = Generator(nr_input_var=3, nr_filter=16)
    summary(model, (1,3))
    if write_to_file:
        text_file = open(file_name, "w")
        text_file.write(str(summary(model, (1,3))))
        text_file.close()

    return None

if __name__ == '__main__':
    GetModelSummary(write_to_file=False)