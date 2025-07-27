import re

# Read the file
with open('src/TensorLinearAlgebra.cpp', 'r') as f:
    content = f.read()

# Define replacements
replacements = [
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<float>::getrf_bufferSize\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnSgetrf_bufferSize");')),
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<double>::getrf_bufferSize\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnDgetrf_bufferSize");')),
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<float>::getrf\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnSgetrf");')),
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<double>::getrf\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnDgetrf");')),
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<float>::getrs\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnSgetrs");')),
    (r'cusolver_utils::cusolver_check\(\s*cusolver_utils::CusolverTraits<double>::getrs\([^)]+\)\s*\);', 
     lambda m: m.group(0).replace(');', ', "cusolverDnDgetrs");')),
]

# Apply simple replacements first
content = content.replace(
    'cusolver_utils::CusolverTraits<float>::getrf_bufferSize(\n                        handle, static_cast<int>(n), static_cast<int>(n),\n                        a_work->data(), static_cast<int>(n), &lwork\n                    )\n                );',
    'cusolver_utils::CusolverTraits<float>::getrf_bufferSize(\n                        handle, static_cast<int>(n), static_cast<int>(n),\n                        a_work->data(), static_cast<int>(n), &lwork\n                    ), "cusolverDnSgetrf_bufferSize"\n                );'
)

content = content.replace(
    'cusolver_utils::CusolverTraits<double>::getrf_bufferSize(\n                        handle, static_cast<int>(n), static_cast<int>(n), \n                        a_work->data(), static_cast<int>(n), &lwork\n                    )\n                );',
    'cusolver_utils::CusolverTraits<double>::getrf_bufferSize(\n                        handle, static_cast<int>(n), static_cast<int>(n), \n                        a_work->data(), static_cast<int>(n), &lwork\n                    ), "cusolverDnDgetrf_bufferSize"\n                );'
)

# Fix the remaining calls by pattern matching
patterns = [
    (r'(\s+cusolver_utils::cusolver_check\(\s+cusolver_utils::CusolverTraits<float>::getrf\([^)]+\)\s+\);)', r'\1'.replace(');', ', "cusolverDnSgetrf");')),
    (r'(\s+cusolver_utils::cusolver_check\(\s+cusolver_utils::CusolverTraits<double>::getrf\([^)]+\)\s+\);)', r'\1'.replace(');', ', "cusolverDnDgetrf");')),
    (r'(\s+cusolver_utils::cusolver_check\(\s+cusolver_utils::CusolverTraits<float>::getrs\([^)]+\)\s+\);)', r'\1'.replace(');', ', "cusolverDnSgetrs");')),
    (r'(\s+cusolver_utils::cusolver_check\(\s+cusolver_utils::CusolverTraits<double>::getrs\([^)]+\)\s+\);)', r'\1'.replace(');', ', "cusolverDnDgetrs");')),
]

# Write the file back
with open('src/TensorLinearAlgebra.cpp', 'w') as f:
    f.write(content)

print("Fixed cusolver_check calls")
