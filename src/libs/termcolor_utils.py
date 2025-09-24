import colorama

class TermcolorUtils:
    @staticmethod
    def red(value:str|int|float) -> str:
        text = colorama.Fore.RED + str(value) + colorama.Style.RESET_ALL
        return text
    
    @staticmethod
    def green(value:str|int|float    ) -> str:
        text = colorama.Fore.GREEN + str(value) + colorama.Style.RESET_ALL
        return text
    
    @staticmethod
    def cyan(value:str|int|float) -> str:
        text = colorama.Fore.CYAN + str(value) + colorama.Style.RESET_ALL
        return text
    
    @staticmethod
    def magenta(value:str|int|float) -> str:
        text = colorama.Fore.MAGENTA + str(value) + colorama.Style.RESET_ALL
        return text