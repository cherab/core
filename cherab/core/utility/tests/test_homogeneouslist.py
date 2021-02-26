import unittest

from cherab.core.utility import HomogeneousList


class TestHomogeneousList(unittest.TestCase):
    
    def test_init(self):

        
        with self.assertRaises(TypeError):
            hlist = HomogeneousList(open)
        
        hlist = HomogeneousList(int)
        
        self.assertEqual(hlist.length, 0)
        self.assertEqual(hlist.get_length(), 0)

    def test_add(self):
        
        int_list = [1, 2, 3, 4, 5]
        str_list = ["a", "b", "c"]
        
        hlist = HomogeneousList(str)

        with self.assertRaises(TypeError):
            hlist.add(1)
        
        hlist.add("a")
        self.assertEqual(hlist.length, 1)
        self.assertEqual(hlist.get_length(), 1)
        self.assertEqual(hlist[0], "a")

        hlist.add("b")
        self.assertEqual(hlist.length, 2)
        self.assertEqual(hlist.get_length(), 2)
        self.assertEqual(hlist[1], "b")
        
    def test_set(self):
        
        int_list = [1, 2, 3, 4, 5]
        str_list = ["a", "b", "c"]
        
        hlist = HomogeneousList(str)

        with self.assertRaises(TypeError):
            hlist.set(int_list)
            
        hlist.set(str_list)
        self.assertEqual(hlist.get_list(), str_list)
    
    def test_remove(self):

        str_list = ["a", "b", "c"]
        hlist = HomogeneousList(str)
        hlist.set(str_list)

        hlist.remove("b")
        self.assertEqual(hlist.get_list(), ["a", "c"])
        self.assertEqual(hlist.length, 2)

    def test_clear(self):

        str_list = ["a", "b", "c"]
        hlist = HomogeneousList(str)
        hlist.set(str_list)

        hlist.clear()
        self.assertEqual(hlist.length, 0)

        