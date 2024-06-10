VERBOSE="-v"

# 스크립트에 전달될 나머지 고정 인수들
FIXED_ARGS="-n 1 1024 1024 1024"

# 스레드 수를 1부터 256까지 더블링하여 실행
for (( t=1; t<=256; t*=2 ))
do
    echo "Running with $t threads..."
    ./run.sh $VERBOSE -n $t $FIXED_ARGS
done