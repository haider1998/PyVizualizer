# examples/sample_project/module_b.py

from module_a import ClassA


class ClassB:
    def method_b(self):
        print("ClassB: method_b called")

    def method_b_with_a(self, a: ClassA):
        print("ClassB: method_b_with_a called")
        a.method_a()  # Calling method from ClassA
