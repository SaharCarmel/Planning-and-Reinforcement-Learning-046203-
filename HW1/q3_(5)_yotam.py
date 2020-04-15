transition_cost = [[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]]
actions = {
    'B': 0,
    'K': 1,
    'O': 2,
    '-': 3,
}


def dp_sol(k):
    word = 'B'
    current_state = 'B'

    for i in range(k):


    get_best_action()

    return word


print(dp_sol(5))

# import math
#
# def mostProbableWord(K,P):
#     C=math.log(1./P)
#     hist=['B','K','O']
#     charWord= K * [0]
#     temp=[float('inf')]* K #1=b,2=k,3=o
#     Vk = [[temp],[temp],[temp]]
#     for stage in reversed(range(K)):
#         if stage==K: #that means we are at the stage before '-':
#             for letter in range(3):
#                 Vk[letter][K]=C(letter,4) math.log(letter)
#
#         elif stage!=1 and stage!=K:
#             for letter=1:3
#                 Vk(letter,stage)=min(C(letter,1:3)+Vk(1:3,stage+1)')
#
#         else: # means stage=1
#             Vk(1,stage)=min(C(1,1:3)+Vk(1:3,stage+1)')
#         #the other two letters in the first stage are ignored !
#
#
# charWord(1)=hist(1)
# for stage=2:K
#  if stage==2
#  [p,index]=min(C(1,1:3)+Vk(1:3,stage)')
#  charWord(stage)=hist(index)
#  prev_index=index
#  else
#  [p,index]=min(C(prev_index,1:3)+Vk(1:3,stage)')
#  charWord(stage)=hist(index)
#  prev_index=index
#  end
# end
# word=string(charWord)