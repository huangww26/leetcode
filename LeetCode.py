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
