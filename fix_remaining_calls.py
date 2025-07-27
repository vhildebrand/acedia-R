import re

with open('src/TensorLinearAlgebra.cpp', 'r') as f:
    content = f.read()

# Fix patterns where the function name is missing
fixes = [
    # Double getrf_bufferSize calls (remove duplicates)
    (r'(\s+), "cusolverDnDgetrf_bufferSize"\s+, "cusolverDnDgetrf_bufferSize"\);', r'\1);'),
    (r'(\s+), "cusolverDnSgetrf_bufferSize"\s+, "cusolverDnSgetrf_bufferSize"\);', r'\1);'),
    
    # Add missing function names
    (r'(cusolver_utils::CusolverTraits<double>::getrf\([^)]+info\s*)\s*\)\s*\);', r'\1), "cusolverDnDgetrf");'),
    (r'(cusolver_utils::CusolverTraits<double>::getrs\([^)]+info\s*)\s*\)\s*\);', r'\1), "cusolverDnDgetrs");'),
]

for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE|re.DOTALL)

with open('src/TensorLinearAlgebra.cpp', 'w') as f:
    f.write(content)

print("Fixed remaining cusolver calls")
