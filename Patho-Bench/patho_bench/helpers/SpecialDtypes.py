"""
This class contains custom dtypes used when parsing arguments.
"""

class SpecialDtypes:
    @staticmethod
    def none_or_str(s):
        '''
        Convert string to None if string is 'None', otherwise return string.
        '''
        return None if s == 'None' else s
    
    @staticmethod
    def none_or_int(s):
        '''
        Convert string to None if string is 'None', otherwise return int.
        '''
        return None if s == 'None' else int(s)
    
    @staticmethod
    def none_or_float(s):
        '''
        Convert string to None if string is 'None', otherwise return float.
        '''
        return None if s == 'None' else float(s)
    
    @staticmethod
    def bool(s):
        '''
        Convert string to bool.
        '''
        if s.lower() not in ['true', 'false']:
            raise ValueError(f"Invalid boolean value: {s}")
        return s.lower() == 'true'
    
    @staticmethod
    def float_or_adaptive(s):
        '''
        Convert string to float if string is not 'adaptive' or 'gridsearch', otherwise return string.
        '''
        return float(s) if s not in ['adaptive', 'gridsearch'] else s