# 2. Add Two Numbers
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        p, q= l1, l2
        carry = 0
        r = p
        while p and q:
            _sum = p.val + q.val + carry
            p.val = _sum % 10
            carry = _sum / 10
            if not p.next:
                r = p
            p, q = p.next, q.next

        if q:
            r.next = q
            p = q
        while p:
            _sum = p.val + carry
            p.val = _sum % 10
            carry = _sum / 10
            if not p.next:
                r = p
            p = p.next
            if _sum < 10:
                break
        if carry > 0:
            r.next = ListNode(carry)

        return l1

    def addTwoNumbers2(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        p = dummy = ListNode(0)
        carry = 0
        while l1 or l2:
            val = carry + (l1.val if l1 else 0) + (l2.val if l2 else 0)
            carry = val / 10
            val %= 10
            p.next = ListNode(val)
            p = p.next
            l1 = l1 and l1.next
            l2 = l2 and l2.next
        if carry > 0:
            p.next = ListNode(carry)
            p = p.next

        return dummy.next
# ---------------------------------------------------------------------- 

# 23. Merge k Sorted Lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if not lists:
            return None
        n = len(lists)
        
        while n > 1:
            k = (n + 1) >> 1
            for i in range(n >> 1):
                lists[i] = self.mergeTwoLists(lists[i], lists[i + k])
            n = k
        return lists[0]
    
    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(0)
        p = dummy

        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                p = l1
                l1 = l1.next
            else:
                p.next = l2
                p = l2
                l2 = l2.next
        if l1:
            p.next = l1
        elif l2:
            p.next = l2

        return dummy.next
# ----------------------------------------------------------------------       

# 24. Swap Nodes in Pairs
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        p = dummy = ListNode(0)
        p.next = head
        left = head
        right = left.next
        while left and right:
            left.next = right.next
            right.next = left
            p.next = right

            p = left
            left = left.next
            if left:
                right = left.next
        return dummy.next

# ----------------------------------------------------------------------

# 61. Rotate List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return head

        total = 1
        p = head
        while p.next:
            total += 1
            p = p.next

        k %= total
        if k == 0:
            return head

        cnt = total - k
        p.next = head
        while cnt > 0:
            p = p.next
            cnt -= 1
        head = p.next
        p.next = None

        return head
# ----------------------------------------------------------------------

# 82. Remove Duplicates from Sorted List II
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        res = ListNode(0)
        tail = res

        isRepeated = False
        while head: 
            while  head.next and head.val == head.next.val:
                isRepeated = True
                head = head.next
            if isRepeated:
                isRepeated = False
            else:
                tail.next = head
                tail = tail.next
            head = head.next
        tail.next = None
        return res.next

    def deleteDuplicates2(self,head):
        if not head or not head.next:
            return head
        p = ans = ListNode(0)
        flag = False
        while head:
            while head.next and head.val == head.next.val:
                head = head.next
                flag = True
            if flag:
                p.next = head.next
                flag = False
            else:
                p = p.next
                head = head.next
        return ans.next
        
    def deleteDuplicates3(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        p = head

        while p and p.next:
            if p.val != p.next.val:
                prev = p 
                p = p.next
            else:
                val = p.val
                n = p.next.next
                while n:
                    if n.val != val:
                        break
                    n = n.next
                prev.next = n
                p = n

        return dummy.next
# ----------------------------------------------------------------------

# 83. Remove Duplicates from Sorted List 
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        p = head

        while p and p.next:
            if p.val == p.next.val:
                p.next = p.next.next
            else:
                p = p.next
        return head
# ----------------------------------------------------------------------

# 86. Partition List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        p = q = dummy
        
        while p.next and p.next.val < x:
            p = p.next
            q = q.next
            
        while q and q.next:
            while q and q.next and q.next.val >= x:
                q = q.next

            tmp = q.next
            if not tmp:
                break
            q.next = q.next.next

            tmp.next = p.next
            p.next = tmp
            p = tmp
        return dummy.next

    def partition2(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        less = ListNode(0)
        greater = ListNode(0)
        p, q, r = less, greater, head

        while r:
            if r.val < x:
                p.next = r
                p = r
            else:
                q.next = r
                q = r
            r = r.next
        
        q.next = None
        p.next = greater.next

        return less.next
# ----------------------------------------------------------------------

# 92. Reverse Linked List II
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        r = dummy
        p = head
        while m > 1:
            r, p = r.next, p.next
            m, n = m - 1, n - 1
        q = p.next
        while n > 1:
            tmp = q.next
            q.next = p
            p = q
            q = tmp
            n -= 1
        r.next.next = q
        r.next = p

        return dummy.next

    def reverseBetween2(self, head, m, n):
        dummy = ListNode(0)
        p = dummy
        while m > 1:
            p.next = head
            p, head = p.next, head.next
            m, n = m - 1, n - 1
        tmp = head
        while n > 1:
            q = head.next
            head.next = p.next
            p.next =head
            head = q
            n -= 1
        tmp.next = head
        return dummy.next
# ----------------------------------------------------------------------

# 94. Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        s = []
        res = []
        p = root
        while p or s:
            while p:
                s.append(p)
                p = p.left
            if s:
                p = s.pop()
                res.append(p.val)
                p = p.right
        return res
    def inorderTraversal2(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        cur = root
        while cur:
            if not cur.left:
                res.append(cur.val)
                cur = cur.right
            else:
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    pre.right = cur
                    cur = cur.left
                else:
                    pre.right = None
                    res.append(cur.val)
                    cur = cur.right
        return res
# ---------------------------------------------------------------------- 

# 95. Unique Binary Search Trees II
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n == 0:
            return []
        values = [i+1 for i in range(n)]     
        return self.helper(range())

    def helper(self, A):
        if len(A) == 0:
            return [None]
        res = []
        for index, val in enumerate(A):
            leftSubTree = self.helper(A[:index])
            rightSubTree = self.helper(A[index+1:])
            for leftChild in leftSubTree:
                for rightChild in rightSubTree:
                    node = TreeNode(val)
                    node.left = leftChild
                    node.right = rightChild
                    res.append(node)
        return res
# ----------------------------------------------------------------------

# 96. Unique Binary Search Trees
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        num = [0] * (n+1)
        num[0] = 1
        for i in range(1, n+1):
            for j in range(i):
                num[i] += num[j] * num[i-1-j]
        return num[n]
# ----------------------------------------------------------------------

# 98. Validate Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    pre = None
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        if not isValidBST(root.left):
            return False
        if self.pre and self.pre.val >= root.val:
            return False
        self.pre = root
        return slef.isValidBST(root.right)
    def isValidBST2(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        pre = None
        s = []
        while True:
            while root:
                s.append(root)
                root = root.left
            if not s:
                return True
            node = s.pop()
            if pre and pre.val >= node.val:
                return False
            pre = node
            root = node.right
# ----------------------------------------------------------------------

# 99. Recover Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        cur, pre, first, second, parent = root, None, None, None, None
        while cur:
            if not cur.left:
                if parent and parent.val > cur.val:
                    if not first:
                        first = parent
                    second = cur
                parent = cur
                cur = cur.right
            else:
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    pre.right = cur
                    cur = cur.left
                else:
                    if parent and parent.val > cur.val:
                        if not first:
                            first = parent
                        second = cur
                    parent = cur
                    pre.right = None
                    cur = cur.right
        if first and second:
            first.val, second.val = second.val, first.val
# ----------------------------------------------------------------------

# 100. Same Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        if p and q and p.val == q.val and \
                self.isSameTree(p.left, q.left) and \
                self.isSameTree(p.right, q.right):
                    return True
        return False

    def isSameTree2(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        s1, s2 = [p], [q]
        while s1:
            r, s = s1.pop(), s2.pop()
            if (not r or not s) and r != s:
                return False
            elif r and s:
                if r.val != s.val:
                    return False
                elif r.left or r.right:
                    s1.append(r.right)
                    s1.append(r.left)
                    s2.append(s.right)
                    s2.append(s.left)
        return True
# ----------------------------------------------------------------------

# 101. Symmetric Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        left_queue, right_queue = [root.left], [root.right]
        while left_queue or right_queue:
            p, q = left_queue.pop(0), right_queue.pop(0)
            if not p or not q:
                if p != q:
                    return False
            else:
                if p.val != q.val:
                    return False
                if p.left or p.right:
                    left_queue.append(p.left)
                    left_queue.append(p.right)
                    right_queue.append(q.right)
                    right_queue.append(q.left)
        return True
    def isSymmetric2(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.isReverse(root.left, root.right)
    def isReverse(self, p, q):
        if not p or not q:
            if p != q:
                return False
            else:
                return True
        if p.val != q.val:
            return False
        return self.isReverse(p.left, q.right) and self.isReverse(p.right, q.left)

# ----------------------------------------------------------------------

# 102. Binary Tree Level Order Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        s = [root]
        while s:
            count = len(s)
            tmp = []
            while count > 0:
                p = s.pop(0)
                tmp.append(p.val)
                if p.left:
                    s.append(p.left)
                if p.right:
                    s.append(p.right)
                count -= 1
            res.append(tmp)
        return res
    
    def levelOrder2(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        if not root:
            return res
        self.addNode(root, res, 0)
        return res
    def addNode(self, node, res, count):
        if not node:
            return
        if len(res) <= count:
            tmp = [node.val]
            res.append(tmp)
        else:
            res[count].append(node.val)
        self.addNode(node.left, res, count+1)
        self.addNode(node.right, res, count+1)
# ---------------------------------------------------------------------- 

# 103. Binary Tree Zigzag Level Order Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        s = [root]
        toLeft = True
        while s:
            count = len(s)
            tmp = []
            while count > 0:
                p = s.pop(0)
                if toLeft:
                    tmp.append(p.val)
                else:
                    tmp.insert(0, p.val)
                if p.left:
                    s.append(p.left)
                if p.right:
                    s.append(p.right)
                count -= 1
            res.append(tmp)
            toLeft = False if toLeft else True
        return res
# ----------------------------------------------------------------------

# 104. Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        max_depth = 0
        if not root:
            return max_depth
        s = [root]
        while s:
            count = len(s)
            for _ in range(count):
                tmp = s.pop(0)
                if tmp.left:
                    s.append(tmp.left)
                if tmp.right:
                    s.append(tmp.right)
            max_depth += 1
        return max_depth
    def maxDepth2(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth2(root.left)+1, self.maxDepth2(root.right)+1)
# ----------------------------------------------------------------------

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        return self.createBinaryTree(preorder, inorder, len(preorder))
    def createBinaryTree(self, preorder, inorder, n):
        if n == 0:
            return None
        i = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.createBinaryTree(preorder[1:], inorder, i)
        root.right = self.createBinaryTree(preorder[i+1:], inorder[i+1:], n-i-1)

        return root
    
    def buildTree2(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        s = [root]
        for i in preorder[1:]:
            tmp = TreeNode(i)
            if inorder.index(i) < inorder.index(s[-1].val):
                s[-1].left = tmp
            else:
                pre = s.pop()
                while s and inorder.index(i) > inorder.index(s[-1].val):
                    pre = s.pop()
                pre.right = tmp
            s.append(tmp)
        return root
# ----------------------------------------------------------------------

# 106. Construct Binary Tree from Inorder and Postorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        return self.createBinaryTree(inorder, postorder, len(inorder)-1, 0, len(postorder)-1)
    def createBinaryTree(self, inorder, postorder, index, start, end):
        if index < 0 or start > end:
            return None
        i = inorder.index(postorder[index])
        root = TreeNode(postorder[index])
        root.left = self.createBinaryTree(inorder, postorder, index-(end-i+1), start, i-1)
        root.right = self.createBinaryTree(inorder, postorder, index-1, i+1, end)
        return root
# ----------------------------------------------------------------------

# 107. Binary Tree Level Order Traversal II
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        s = [root]
        while s:
            count = len(s)
            tmp = []
            while count > 0:
                p = s.pop(0)
                tmp.append(p.val)
                if p.left:
                    s.append(p.left)
                if p.right:
                    s.append(p.right)
                count -= 1
            res.insert(0, tmp)
        return res
# ----------------------------------------------------------------------

# 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.createBST(nums, 0, len(nums))
    def createBST(self, A, start, end):
        if start >= end:
            return None
        mid = (start + end) / 2
        if (start + end) % 2 != 0:
            mid += 1
        root = TreeNode(A[mid])
        root.left = self.createBST(A, start, mid-1)
        root.right = self.createBST(A, mid+1, end)
        return root
# ----------------------------------------------------------------------

# 109. Convert Sorted List to Binary Search Tree
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        slow, fast = head, head.next.next
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        root = TreeNode(slow.next.val)
        root.right = self.sortedListToBST(slow.next.next)
        slow.next = None
        root.left = self.sortedListToBST(head)
        return root
# ----------------------------------------------------------------------

# 110. Balanced Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        left_height = self.height(root.left)
        right_height = self.height(root.right)
        if abs(left_height - right_height) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
    def height(self, root):
        if not root:
            return 0
        lh = self.height(root.left)
        rh = self.height(root.right)
        return max(lh, rh)+1

    def isBalanced2(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.dfs(root) != -1
    def dfs(self, root):
        if not root:
            return 0
        lh = self.dfs(root.left)
        if lh == -1:
            return -1
        rh = self.dfs(root.right)
        if rh == -1:
            return -1
        return -1 if abs(lh - rh) > 1 else max(lh, rh)+1
# ----------------------------------------------------------------------

# 111. Minimum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left:
            return self.minDepth(root.right)
        elif not root.right:
            return self.minDepth(root.left)
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right))+1
# ----------------------------------------------------------------------

# 114. Flatten Binary Tree to Linked List 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        if root:
            self.flatten(root.left)
            self.flatten(root.right)
            
            p = dummy = TreeNode(0)
            dummy.right = root.left
            while p.right:
                p = p.right
            p.right = root.right

            root.left = None
            root.right = dummy.right
# ----------------------------------------------------------------------

# 138. Copy List with Random Pointer
# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        p = dummy = RandomListNode(0)
        q = head
        m = {}
        while q:
            tmp = RandomListNode(q.label)
            p.next = tmp
            p = tmp
            tmp.random = q.random

            m[q] = tmp

            q = q.next
        q = dummy.next
        while q:
            if q.random:
                q.random = m[q.random]
            q = q.next
            
        return dummy.next

    def copyRandomList2(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if not head:
            return None

        # 遍历并插入新节点
        h = head
        while h:
            node = RandomListNode(h.label)
            node.random = h.random

            tmp = h.next
            h.next = node
            node.next = tmp
            h = tmp

        # 调整random
        h = head.next
        while h:
            if h.random:
                h.random = h.random.next
            if not h.next:
                break
            h = h.next.next

        # 断开链表
        h = head
        dummy = p = RandomListNode(0)
        while h:
            p.next = h.next
            p = p.next
            h.next = h.next.next
            h = h.next

        return dummy.next
# ----------------------------------------------------------------------

# 141. Linked List Cyc# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = fast = head

        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                return True

        return slow == fast
# ----------------------------------------------------------------------

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

        myHead = ListNode(0)
        p = myHead
        while curPtr:
            nextPtr = curPtr.next
            if myHead.next and myHead.next.val > curPtr.val or p.next and curPtr.val < p.next.val:
                p = myHead
            while p.next and p.next.val < curPtr.val:
                p = p.next

            curPtr.next = p.next
            p.next = curPtr

            curPtr = nextPtr

        return myHead.next

    def insertionSortList2(self, head):
        p = tmp = ListNode(0)
        cur = tmp.next = head
        while cur and cur.next:
            val = cur.next.val
            if cur.val < val:
                cur = cur.next
                continue
            if p.next.val > val:
                p = tmp
            while p.next.val < val:
                p = p.next
            new = cur.next
            cur.next = new.next
            new.next = p.next
            p.next = new
        return tmp.next
# ----------------------------------------------------------------------

# 148. Sort List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        slow = fast = head
        while fast and fast.next:
            fast =fast.next
            if slow.next != fast:
                slow = slow.next
            if fast:
                fast = fast.next
        p = slow.next
        slow.next = None
        head = self.sortList(head)
        p = self.sortList(p)
        head = self.merge(head, p)

        return head

    def merge(self, head1, head2):
        head = ListNode(0)
        cur = head
        while head1 and head2:
            if head1.val < head2.val:
                cur.next = head1
                cur = cur.next
                head1 = head1.next
            else:
                cur.next = head2
                cur = cur.next
                head2 = head2.next
        while head1:
            cur.next = head1
            cur = cur.next
            head1 = head1.next
        while head2:
            cur.next = head2
            cur = cur.next
            head2 = head2.next
        return head.next
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

# 203. Remove Linked List Elements# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        p = dummy = ListNode(0)
        p.next = head
        while p and p.next:
            if p.next.val == val:
                p.next = p.next.next     
            else:
                p = p.next

        return dummy.next
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

# 234. Palindrome Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = fast = head
        pre = None

        while fast and fast.next:
            tmp = slow
            slow = slow.next
            fast = fast.next.next
            tmp.next = pre
            pre = tmp
        if fast:
            slow = slow.next
        while slow and pre.val == slow.val:
            slow, pre = slow.next, pre.next

        return slow == None
# ----------------------------------------------------------------------

# 237. Delete Node in a Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node.next:
            node.val = node.next.val
            node.next = node.next.next
        else:
            node = None
# ----------------------------------------------------------------------

# 328. Odd Even Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        odd, even = head, head.next
        p = even

        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = p

        return head
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 341. Flatten Nested List Iterator
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.S = []
        self.nextItem = 0
        for i in range(len(nestedList)-1, -1, -1):
            self.S.append(nestedList[i])

    def next(self):
        """
        :rtype: int
        """
        return self.nextItem

    def hasNext(self):
        """
        :rtype: bool
        """
        while self.S:
            curItem = self.S.pop()
            if curItem.isInteger():
                self.nextItem = curItem
                return True
            nextList = curItem.getList()
            for i in range(len(nextList)-1, -1, -1):
                self.S.append(nextList[i])
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
# ----------------------------------------------------------------------
