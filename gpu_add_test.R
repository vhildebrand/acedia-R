# gpu_add_test.R


# shared lib name
lib_file <- "gpu_add.so"

# load the shared library
dyn.load(lib_file)

gpu_add <- function(a, b) {
    # The .Call function takes the name of the C function in the library
    # and the args to pass to it
    .Call("r_gpu_add", a, b)
}

n <- 10e6
vec1 <- runif(n)
vec2 <- runif(n)


cpu_result <- vec1 + vec2


gpu_result <- gpu_add(vec1, vec2)



all.equal(cpu_result, gpu_result)


library(microbenchmark)


microbenchmark(
    cpu = vec1 + vec2,
    gpu = gpu_add(vec1, vec2),
    times = 10
)



# unload the lib when done?
dyn.unload(lib_file)
