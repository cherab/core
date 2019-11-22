
from cherab.openadas import OpenADAS, print_available_adf15_rates


adas = OpenADAS()

print_available_adf15_rates("adf15/pec96#c/pec96#c_vsu#c1.dat")

print()
print()

print_available_adf15_rates("pec96#he_pju#he0.dat")

print()
print()

print_available_adf15_rates("pec96#he_pju#he1.dat")

print()
print()

print_available_adf15_rates("pec12#h_pju#h0.dat")
