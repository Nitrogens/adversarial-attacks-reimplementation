class A(object):
    def __init__(self):
        self.a = 1
        self.b = 2
    
    def opt(self):
        print(self.a + self.b)

class B(A):
    def __init__(self):
        super().__init__()
    
    def test(self):
        super().opt()
        self.b = 3
        self.a = 2
        super().opt()

B_1 = B()
B_1.test()