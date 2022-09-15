# # import module sys to get the type of exception
# import sys
#
# randomList = ['a', 0, 2]
#
# for entry in randomList:
#     try:
#         print("The entry is", entry)
#         r = 1/int(entry)
#         break
#     except:
#         print("Oops!", sys.exc_info()[0], "occurred.")
#         print("Next entry.")
#         print()
# print("The reciprocal of", entry, "is", r)

# import module sys to get the type of exception
import sys

#----------------------------------------------------------
# Handling exception as e
#
# randomList = ['a', 0, 2]
#
# for entry in randomList:
#     try:
#         print("The entry is", entry)
#         r = 1/int(entry)
#         break
#     except Exception as e:
#         print("Oops!", e.__class__, "occurred.")
#         print("Next entry.")
#         print()
# print("The reciprocal of", entry, "is", r)

#----------------------------------------------------------

# #Handling special types of exceptions in different ways
# try:
#    # do something
#    pass
#
# except ValueError:
#    # handle ValueError exception
#    pass
#
# except (TypeError, ZeroDivisionError):
#    # handle multiple exceptions
#    # TypeError and ZeroDivisionError
#    pass
#
# except:
#    # handle all other exceptions
#    pass

#----------------------------------------------------------
# Raising exception by programmer in python
# try:
#      a = int(input("Enter a positive integer: "))
#      if a <= 0:
#          raise ValueError("That is not a positive number!")
# except ValueError as ve:
#      print(ve)

# program to print the reciprocal of even numbers

#----------------------------------------------------------
# Try--except--else clause for handling errors
# try:
#     num = int(input("Enter a number: "))
#     #assert is another way of raising an exception (AssertionError) based on cutom condition, finally that exception will be handled by except
#     assert num != 0
# except:
#     print("Not an even number!")
# else:
#     reciprocal = 1/num
#     print(reciprocal)

#----------------------------------------------------------
#Another example based on AssertionError
# def square(x):
#     assert x>=0, 'Only positive numbers are allowed'
#     return x*x
#
# try:
#     square(-2)
# except AssertionError as msg:
#     print(msg)

