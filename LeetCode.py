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
