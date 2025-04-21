import unittest
import numpy as np
from homography_matrix import homography_matrix_v2

class MyTestCase(unittest.TestCase):
    def matrix_test(self): #todo cleanup needed

        # homography_matrix_v2(np.array([10.,0.]),np.array([-10.,0.]),(100,100))
        for i in range(1, 5):
            center = np.array([100, 100])
            offset = np.array([0, 10])
            homography_matrix_v2(np.array([10. * i, 0.]) + center + offset,
                                 np.array([-10. * i, 0.]) + center + offset, center)
        # print("-"*200)
        # for i in range(1,5):#ok
        #     center = np.array([100,100])*i
        #     offset = np.array([0,10])
        #     homography_matrix_v2(np.array([10.,0.])+offset+center,np.array([-10.,0.])+offset+center,center)
        # print("-"*200)   #ok,same as previous
        # for i in range(1,5):
        #     center = np.array([100,100])*i
        #     offset = np.array([0,10])
        #     homography_matrix_v2(np.array([-10.,0.])+offset+center,np.array([10.,0.])+offset+center,center)
        # print("-"*200)   #ok, same as previous
        # for i in range(1,5):
        #     center = np.array([100,100])*i
        #     offset = np.array([10,0])
        #     homography_matrix_v2(np.array([0.,10.])+offset+center,np.array([0.,-10.])+offset+center,center)
        print(
            "-" * 200)  # this test is failing because my code now works as it should and it cannot process perfectly horizontal camera position
        for i in range(1, 5):
            center = np.array([100, 100])
            offset = np.array([0, 0])
            homography_matrix_v2(np.array([10. * i, 0.]) + center + offset,
                                 np.array([-10. * i, 0.]) + center + offset, center)
        print("-" * 200)
        for i in range(-3, 3):
            center = np.array([100, 100])
            offset = np.array([0, 10]) * i
            homography_matrix_v2(np.array([100., 0.]) + offset + center, np.array([-100., 0.]) + offset + center,
                                 center)
        # print("-"*200)
        # for i in range(1,5): #ok
        #     center = np.array([100,100])*i
        #     offset = np.array([0,10])
        #     homography_matrix_v2(np.array([-10.,0.])+offset+center,np.array([10.,0.])+offset+center,center)
        # print("-"*200)
        # for i in range(1,5): #ok
        #     center = np.array([100,100])*i
        #     offset = np.array([0,-10])
        #     homography_matrix_v2(np.array([-10.,0.])+offset+center,np.array([10.,0.])+offset+center,center)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
