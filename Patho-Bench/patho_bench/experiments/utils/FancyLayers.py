from torch.nn import Module

"""
Running or importing this file will override the default __repr__ method for all torch.nn.Module objects to print the model architecture with emojis indicating whether each layer is trainable or not.
"""

def __repr__(self):
    def _addindent(s_, numSpaces):
        '''
        This internal function helps in adding indentation to multi-line strings.

        Args:
            s_ (str): The string to add indentation to
            numSpaces (int): The number of spaces to indent by
        '''
        # Split the input string by newline to process each line.
        s = s_.split('\n')
        
        # If it's a single line, just return the string as is.
        if len(s) == 1:
            return s_
        
        # For multi-line strings: remove the first line, 
        # add spaces to the beginning of each subsequent line, 
        # and then recombine them into a single string.
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s
    
    def is_trainable(module):
        '''
        Check the requires_grad status for all parameters in the module.
        '''
        try:
            params = list(module.parameters())
            # If the module has no parameters, treat it as neither trainable nor frozen.
            if not params:
                return None
        except:
            # If the module has no parameters, treat it as neither trainable nor frozen.
            return None
        # If all parameters are trainable, return True. If all are frozen, return False.
        requires_grads = [p.requires_grad for p in params]
        if all(requires_grads):
            return True
        elif not any(requires_grads):
            return False
        return None
    
    # Start building the representation string of the object.
    # Start with any extra info specific to this instance (possibly overridden in subclasses).
    extra_lines = []
    extra_repr = self.extra_repr()
    
    # If there's any extra info, split it into separate lines.
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    
    # Collect string representations of child modules (sub-modules contained in this module).
    child_lines = []
    for key, module in self._modules.items():
        # Determine the training status of the module.
        trainable = is_trainable(module)
        # Prepend the appropriate emoji.
        prefix = ''
        if trainable == True:
            prefix = '🔥 '
        elif trainable == False:
            prefix = '⛄️ '
        else:
            prefix = '   '
        
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append(prefix + '(' + key + '): ' + mod_str)
    
    # Combine extra info and child modules info.
    lines = extra_lines + child_lines
    
    # Begin the main representation string with the name of the current object.
    main_str = self._get_name() + '('
    
    # Depending on the contents of 'lines', format the string representation.
    if lines:
        # If there's only one extra info line and no child lines, keep it compact.
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        # Otherwise, display each item on a new line with proper indentation.
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
    
    # Close the main representation string.
    main_str += ')'
    
    return main_str

# Override the default __repr__ method for all torch.nn.Module objects.
Module.__repr__ = __repr__