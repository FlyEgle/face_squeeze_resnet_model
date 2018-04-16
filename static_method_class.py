class A(object):

    def instance_method(self, n):
        print('self:', self)

    @classmethod
    def class_method(cls, n):
        print(n)

    @staticmethod
    def static_method(n):
        print(n)



"""
类里面的一个静态方法，跟普通函数没什么区别，与类和实例都没有所谓的绑定关系，它只不过是碰巧存在类中的一个函数而已。不论是通过类还是实例都可以引用该方法。
"""
