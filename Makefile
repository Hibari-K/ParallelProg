CC		= mpiicc
CFLAGS	= -qopenmp -axMIC-AVX512
OBJ1	= nbody
OBJ2	= vec_nbody

EXEC	= mpiexec.hydra
EFLAGS	= -n
PROC	= 32768

all:
	$(CC) $(CFLAGS) $(OBJ1).c -o $(OBJ1)
	$(CC) $(CFLAGS) $(OBJ2).c -o $(OBJ2)

run:
	$(EXEC) $(EFLAGS) $(PROC) ./$(OBJ2)
