import sys

def error_message_details(error,error_details:sys):
    _,_,exe_tab = error_details.exc_info()
    file_name = exe_tab.tb_frame.f_code.co_filename
    error_message = "This error [{0}] line number [{1}] error message [{2}]".format(
        file_name,exe_tab.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_detalis:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details=error_detalis)

    def __str__(self):
        return self.error_message