'''
for (i = 0; i <= 5; i++) {
    System.out.println(i + "값");
}

for 변수명 in 배열들:
    print(변수명)
'''

과일들=["사과", "오렌지", "바나나"]
for f in 과일들:
    print(f)

# 배열이 아니라 특정 숫자 사이 값을 출력해야 할 때
for i in range(1, 6): # 1 ~ 5 까지 출력 range(a, b) ◀ a 이상 ~ b 미만
    print(i)

# 중첩 for 문
for i in range(2, 10):
    for j in range(1, 10):
        print(f"{i} x {j} = {i * j}")
    print() # 단 사이에 빈 줄 하나 추가