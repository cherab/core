# Copyright (c) 2015, Dr Alex Meakins
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. The name of the author may not be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


class RecursiveDict(dict):
    """
    A dictionary that implements a basic, automatically expanding tree.

    If a key is accessed that is not defined then it is automatically populated
    with another RecursiveDict object. This allows the user to rapidly
    construct nested trees of data, with each level of the tree automatically
    created. The RecursiveDict is especially useful for quickly assembling
    configuration files. Once the RecursiveDict is populated it can be frozen by
    converting the tree to a nested set of basic python dictionaries.
    
    For example::

        a = RecursiveDict()
        a["animal"]["bird"]["parrot"]["dead"] = True
        a["tree"]["larch"] = "The larch."
        b = a.freeze()

    This will produce the following nested dictionary in b::

        b = {
            "animal": {
                "bird": {
                    "parrot": {
                        "dead": True
                    }
                }
            },
            "tree": {
                "larch": "The larch."
            }
        }
    """

    def __missing__(self, key):
        """
        Missing keys are automatically populated with RecursiveDicts.
        """

        value = self[key] = type(self)()
        return value

    def freeze(self):
        """
        Returns a copy of this object with the RecursiveDicts replaced with basic python dictionaries.
        """

        d = dict(self)
        for key, value in d.items():
            if isinstance(value, RecursiveDict):
                d[key] = value.freeze()
        return d

    @classmethod
    def from_dict(cls, dictionary):
        """
        Returns a copy of the dictionary as a RecursiveDict.
        """
        return cls._convert_dict_tree(dictionary)

    @classmethod
    def _convert_dict_tree(cls, dict_tree):
        rd = RecursiveDict()
        for key, value in dict_tree.items():
            if isinstance(value, dict):
                value = cls._convert_dict_tree(value)
            rd[key] = value

        return rd
