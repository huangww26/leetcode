# 142. Linked List Cycle II
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        i, j = head, head
        while True:
            if j.next and j.next.next:
                i, j = i.next, j.next.next
                if i == j:
                    i = head
                    while i != j:
                        i, j = i.next, j.next
                    return i
            else:
                return None
# ----------------------------------------------------------------------

# 143. Reorder List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if head and head.next:
            p1 = p2 = head
            while p2.next:
                p1 = p1.next
                p2 = p2.next
                if p2.next:
                    p2 = p2.next
            s = []
            p2 = p1.next
            p1.next = None
            while p2:
                s.append(p2)
                p2 = p2.next
            p1 = head
            while s and p1:
                p2 = s.pop()
                p2.next = p1.next
                p1.next = p2
                p1 = p2.next
            
# ----------------------------------------------------------------------

# 144. Binary Tree Preorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stack = []
        res = []
        stack.append(root)
        while stack:
            p = stack.pop()
            res.append(p.val)
            if p.right: stack.append(p.right)
            if p.left: stack.append(p.left)
        return res
# ----------------------------------------------------------------------

# 145. Binary Tree Postorder Traversal 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        stack1 = []
        stack2 = []
        stack1.append(root)
        while stack1:
            p = stack1.pop()
            stack2.append(p)
            if p.left: stack1.append(p.left)
            if p.right: stack1.append(p.right)
        while stack2:
            res.append(stack2.pop().val)
        return res

    def postorderTraversal2(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        cur = pre = None
        s = [root]
        while s:
            cur = s[-1]
            if (cur.left == None and cur.right == None) or (pre and (pre == cur.left or pre == cur.right)):
                res.append(cur.val)
                s.pop()
                pre = cur
            else:
                if cur.right: s.append(cur.right)
                if cur.left: s.append(cur.left)
         return res
# ----------------------------------------------------------------------

# 147. Insertion Sort List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        curPtr = head
        pHead = pTail = head

        myHead = ListNode(0)

        while curPtr:
            nextPtr = curPtr.next
            p = myHead

            if curPtr.val > pTail.val:
            while p.next and p.next.val < curPtr.val:
                p = p.next

            curPtr.next = p.next
            p.next = curPtr

            curPtr = nextPtr

        return myHead.next
# ----------------------------------------------------------------------

# 160. Intersection of Two Linked Lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pA, pB = headA, headB
        while pA and pB:
            if pA == pB:
                return pA
            else:
                pA, pB = pA.next, pB.next
        if pA:
            while pA:
                pA, headA = pA.next, headA.next
            while headA:
                if headA == headB:
                    return headA
                else:
                    headA, headB = headA.next, headB.next
        else:
            while pB:
                pB, headB = pB.next, headB.next
            while headB:
                if headB == headA:
                    return headB
                else:
                    headA, headB = headA.next, headB.next
                    
    def getIntersectionNode2(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p, q = headA, headB
        while p != q:
            p = p.next if p else headB
            q = q.next if q else headA
        return p

    def getIntersectionNode3(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pA, pB = headA, headB
        lenA = lenB = 0
        while pA or pB:
            if pA:
                lenA += 1
                pA = pA.next
            if pB:
                lenB += 1
                pB = pB.next
        pA, pB = headA, headB
        if lenA > lenB:
            for i in range(lenA - lenB):
                pA = pA.next
        else:
            for i in range(lenB - lenA):
                pB = pB.next
        while pA != pB:
            pA, pB = pA.next, pB.next
        return pA
    
    def getIntersectionNode4(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB:
            return 
        pA, pB = headA, headB
        
        while pB.next:
            pB = pB.next
        dumy = pB
        dumy.next = headB

        res = None
        
        pB = headA
        while pA and pB:
            pA, pB = pA.next, pB.next
            if pB:
                pB = pB.next
            if pA == pB:
                break
        if pA and pB and pA == pB:
            pA = headA
            while pA != pB:
                pA, pB = pA.next, pB.next
            res = pA

        dumy.next = None
        return res
# ----------------------------------------------------------------------

# 206. Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        if head == None:
            return head
        stack = []
        while head:
            stack.append(head)
            head = head.next
        #myhead = ListNode(0)
        head = stack.pop()
        p = head
        while stack:
            p.next = stack.pop()
            p = p.next
        p.next = None    
        return head
# ----------------------------------------------------------------------
