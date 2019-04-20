# if a bulk layer, assume full absorption profile
# also for full RT or full TMM structures
# if mixed, specify for interface layers


if full_RT:
    # need to send a 'group' structure to the RT

elif full_TMM:
    # send OptiStack to TMM

elif full_RCWA:
    # send relevant struct to RCWA solver

elif matrix_method:
    # need to scan through each surface and determine method
