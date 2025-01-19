# examples/sample_project/main.py

from module_a import ClassA
from module_b import ClassB


def main():
    a = ClassA()
    b = ClassB()

    a.method_a()
    b.method_b()
    b.method_b_with_a(a)


if __name__ == "__main__":
    main()
