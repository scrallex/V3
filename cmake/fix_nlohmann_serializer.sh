#!/bin/bash
# Fix nlohmann_json serializer.hpp for GCC 11 compatibility

SERIALIZER_FILE="$1"

if [ -z "$SERIALIZER_FILE" ]; then
    echo "Usage: $0 <path-to-serializer.hpp>"
    exit 1
fi

if [ ! -f "$SERIALIZER_FILE" ]; then
    echo "File not found: $SERIALIZER_FILE"
    exit 1
fi

# Check if already patched
if grep -q "// GCC 11 compatibility patch applied" "$SERIALIZER_FILE"; then
    echo "File already patched"
    exit 0
fi

# Create a backup
cp "$SERIALIZER_FILE" "${SERIALIZER_FILE}.backup"

# Apply the patch using sed
# We need to add 'typename' keyword to help with dependent name lookup
sed -i '50a\
    // GCC 11 compatibility patch applied\
    // Forward declarations for member variables to fix two-phase name lookup issues\
    private:\
    mutable std::array<char, 512> string_buffer{{}};\
    mutable std::array<char, 64> number_buffer{{}};\
    public:' "$SERIALIZER_FILE"

echo "Patched $SERIALIZER_FILE for GCC 11 compatibility"