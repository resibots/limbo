// This file implements 3d hypervolume calculations using an altered
// version of the 2d archiver by Iris Hupkens. 2d hypervolume
// calculations are also included.

// Written by Iris Hupkens, 2013.

#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <deque>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include "ehvi_hvol.h"
using namespace std;

// Some prototypes:
struct avlnode;

struct point {
    // the data for a single point in the set + links to its AVL trees
    double x;
    double y;
    double S;
    avlnode* xnode;
};

struct avlnode {
    // an AVL tree node
    avlnode* left;
    avlnode* right;
    avlnode* parent;
    int balance; // height of left subtree - height of right subtree
    point* data;
};

class bitree {
    // Implementation of the AVL tree
public:
    // stats treestats;
    ~bitree();
    int nodecount; // amount of nodes in the tree
    double xmin, ymin; // coordinates of the reference point
    double totalS; // Hypervolume covered by the approximation set
    avlnode* root; // root for the AVL tree
    bitree(double rx, double ry)
    {
        nodecount = 0;
        totalS = 0;
        xmin = rx;
        ymin = ry;
        root = NULL;
    }
    avlnode* attemptcandidate(double x, double y);
    avlnode* getprevious(avlnode* start);
    avlnode* getnext(avlnode* start);
    void calculateS(avlnode* node);
    point* removenode(avlnode* node);
    void removedominated(avlnode* node);

private:
    void insertrebalance(avlnode* node, int change);
    void deleterebalance(avlnode* node, int change);
    void rotateleft(avlnode* node);
    void rotateright(avlnode* node);
    void recursivedestroyall(avlnode* node);
};

bitree::~bitree()
{
    // Destructor. Prevents memory from leaking all over the place.
    recursivedestroyall(root);
}

void bitree::recursivedestroyall(avlnode* node)
{
    // Destroys the whole tree and everything connected to it.
    if (node == NULL)
        return;
    recursivedestroyall(node->right);
    recursivedestroyall(node->left);
    delete node->data;
    delete node;
}

avlnode* bitree::getprevious(avlnode* start)
{
    // Returns the previous node in the sort order, or NULL if there is no
    // previous node
    avlnode* answer;
    if (start == NULL) {
        return NULL;
    }
    if (start->left) { // return rightmost left child
        answer = start->left;
        while (answer->right != NULL)
            answer = answer->right;
        return answer;
    }
    else {
        answer = start;
        while (answer->parent != NULL && answer->parent->right != answer)
            answer = answer->parent;
        return answer
            ->parent; // return node that start is the leftmost right child of
    }
}

avlnode* bitree::getnext(avlnode* start)
{
    // Returns the next node in the sort order, or NULL if there is no next node
    avlnode* answer;
    if (start == NULL) {
        return NULL;
    }
    if (start->right) { // return leftmost right child
        answer = start->right;
        while (answer->left != NULL) {
            answer = answer->left;
        }
        return answer;
    }
    else {
        answer = start;
        while (answer->parent != NULL && answer->parent->left != answer)
            answer = answer->parent;
        return answer
            ->parent; // return node that start is the rightmost left child of
    }
}

point* bitree::removenode(avlnode* node)
{
    // Remove a node from the tree, returns its data
    int change = 0;
    point* returnvalue;
    if (!node)
        return NULL;
    if (node->parent) {
        if (node->parent->left == node)
            change = -1;
        else
            change = 1;
    }
    if (node->left == NULL) {
        if (node->right == NULL) {
            // leaf node, easy deletion
            if (node->parent == NULL)
                root = NULL;
            else if (node->parent->left == node)
                node->parent->left = NULL;
            else
                node->parent->right = NULL;
        }
        else if (node->parent == NULL) {
            root = node->right;
            node->right->parent = NULL;
        }
        else if (node->parent->left == node) {
            node->parent->left = node->right;
            node->right->parent = node->parent;
        }
        else {
            node->parent->right = node->right;
            node->right->parent = node->parent;
        }
    }
    else if (node->right == NULL) {
        if (node->parent == NULL) {
            root = node->left;
            node->left->parent = NULL;
        }
        else if (node->parent->left == node) {
            node->parent->left = node->left;
            node->left->parent = node->parent;
        }
        else {
            node->parent->right = node->left;
            node->left->parent = node->parent;
        }
    }
    else { // node with two children
        // we replace it with the next node and delete that one.
        // IMPORTANT: this magic trick causes getnext to not
        // work as expected when used in combination with
        // removenode! Always keep that in mind when deleting
        // nodes
        avlnode* next = getnext(node);
        returnvalue = node->data;
        next->data->xnode = node;
        node->data = next->data;
        removenode(next);
        return returnvalue;
    }
    // Everything's set up to actually delete the node, but first
    // we should update the balance information for its parent.
    if (node->parent != NULL) {
        deleterebalance(node->parent, change);
    }
    returnvalue = node->data;
    nodecount--;
    delete node;
    return returnvalue;
}

avlnode* bitree::attemptcandidate(double x, double y)
{
    // Attempt to insert point x, y into the population, return pointer to its
    // associated avlnode on success, NULL on failure.
    avlnode* temp = root;
    avlnode* closest = NULL; // pointers used for all the tree stuff
    // treestats.inserting = true;
    if (!root) {
        // If this is the first candidate, it can always be inserted.
        root = new avlnode;
        root->left = NULL;
        root->right = NULL;
        root->parent = NULL;
        root->balance = 0;
        root->data = new point;
        root->data->x = x;
        root->data->y = y;
        root->data->S = 0;
        root->data->xnode = root;
        nodecount++;
        calculateS(root);
        return root;
    }
    // If the tree isn't empty, make sure the candidate isn't dominated.
    // This can be done by finding the point closest but not lower in x value,
    // and seeing if it is higher in y value.
    bool found = false;
    while (!found) {
        if (temp->data->x < x) {
            if (temp->right != NULL)
                temp = temp->right;
            else
                found = true;
        }
        else if (temp->data->x > x) {
            closest = temp;
            if (temp->left != NULL)
                temp = temp->left;
            else
                found = true;
        }
        else {
            closest = temp;
            found = true;
        }
    }
    if (closest == NULL) { // There is no point with higher x
        temp->right = new avlnode;
        temp->right->left = NULL;
        temp->right->right = NULL;
        temp->right->parent = temp;
        temp->right->balance = 0;
        temp->right->data = new point;
        temp->right->data->x = x;
        temp->right->data->y = y;
        temp->right->data->S = 0;
        temp->right->data->xnode = temp->right;
        temp = temp->right;
        insertrebalance(temp, 0);
    }
    else {
        if (closest->data->y >= y) // Candidate is dominated or the same point
            return NULL;
        // Else candidate isn't dominated. Insert as child of temp.
        if (temp->data->x < x) {
            temp->right = new avlnode;
            temp->right->left = NULL;
            temp->right->right = NULL;
            temp->right->parent = temp;
            temp->right->balance = 0;
            temp->right->data = new point;
            temp->right->data->x = x;
            temp->right->data->y = y;
            temp->right->data->S = 0;
            temp->right->data->xnode = temp->right;
            temp = temp->right;
            insertrebalance(temp, 0);
        }
        else if (temp->data->x > x) {
            temp->left = new avlnode;
            temp->left->left = NULL;
            temp->left->right = NULL;
            temp->left->parent = temp;
            temp->left->balance = 0;
            temp->left->data = new point;
            temp->left->data->x = x;
            temp->left->data->y = y;
            temp->left->data->S = 0;
            temp->left->data->xnode = temp->left;
            temp = temp->left;
            insertrebalance(temp, 0);
        }
        else {
            // Special situation: temp is replaced by the candidate
            temp->data->x = x;
            temp->data->y = y;
            nodecount--;
        }
    }
    nodecount++;
    // We have inserted the point and must now update totalS and remove dominated
    // points.
    calculateS(temp);
    removedominated(temp);
    return temp;
}

void bitree::removedominated(avlnode* node)
{
    // Iterates over points that come after node, stops when it finds one that
    // isn't dominated
    // or reaches the end. First nondominated point's horizontal S strip is
    // recalculated.
    // Dominated points are deleted.
    avlnode* previous = getprevious(node);
    while (previous != NULL && previous->data->y <= node->data->y) {
        avlnode* goner = previous;
        previous = getprevious(previous);
        point* magic;
        if (previous != NULL)
            magic = previous->data; // Protection against the magic trick in
        // removenode
        point* thedata = removenode(goner);
        totalS -= thedata->S;
        delete thedata;
        if (previous != NULL)
            previous = magic->xnode;
    }
    if (previous != NULL)
        calculateS(previous);
}

/*void bitree::showtree(avlnode * node, int depth){
//Display the contents of the tree.
  if (node == NULL)
    return;
  if (node->left)
    showtree(node->left,depth+1);
  for (int i=0;i<depth;i++)
    cout << "-";
  cout << " X: " << node->data->x << " Y: " << node->data->y << " S: " <<
node->data->S
       << " balance: " << node->balance
       << " p: " << (node->parent ? node->parent->data->x : 0)
       << " l: " << (node->left ? node->left->data->x : 0)
       << " r: " << (node->right ? node->right->data->x : 0)
       << endl;
  if(node->right)
    showtree(node->right,depth+1);
}*/

void bitree::calculateS(avlnode* node)
{
    // Recalculates the horizontal S strip for the node (NOT its contribution).
    avlnode* next = getnext(node);
    totalS -= node->data->S;
    node->data->S = node->data->x - xmin;
    if (next == NULL) {
        node->data->S *= node->data->y - ymin;
    }
    else {
        node->data->S *= (node->data->y - next->data->y);
    }
    totalS += node->data->S;
}

void bitree::deleterebalance(avlnode* node, int change)
{
    // Change balance factor of ancestors after deletion.
    // change = -1 for leftdeletion, 1 for rightdeletion.
    if (node == NULL)
        return;
    node->balance += change;
    // If balance is 1 or -1 then everything is fine.
    // We only need to do something if balance is -2, 0 or 2.
    if (node->balance == 0) {
        // subtree has shrunk by 1 so we need to update the parent
        if (node->parent) {
            if (node->parent->left == node)
                deleterebalance(node->parent, -1);
            else
                deleterebalance(node->parent, 1);
        }
    }
    else {
        // Tree is unbalanced. Fix it.
        if (node->balance == 2) {
            if (node->left->balance >= 0) {
                rotateright(node);
            }
            else {
                rotateleft(node->left);
                rotateright(node);
            }
            deleterebalance(node->parent, 0);
        }
        else if (node->balance == -2) {
            if (node->right->balance <= 0) {
                rotateleft(node);
            }
            else {
                rotateright(node->right);
                rotateleft(node);
            }
            deleterebalance(node->parent, 0);
        }
    }
}

void bitree::insertrebalance(avlnode* node, int change)
{
    // Change balance factors and perform necessary rotations.
    // Calling insertrebalance(node,0) on the new node recursively
    // fixes the balance of the whole tree.
    if (change == 0 || node->balance == 0) {
        // node was perfectly balanced
        node->balance = change;
        if (node->parent) {
            if (node->parent->left == node)
                insertrebalance(node->parent, 1);
            else
                insertrebalance(node->parent, -1);
        }
    }
    else {
        node->balance += change;
        // Perform necessary rotations if tree has become unbalanced
        if (node->balance == 2) {
            if (node->left->balance > 0) {
                rotateright(node);
            }
            else {
                rotateleft(node->left);
                rotateright(node);
            }
        }
        else if (node->balance == -2) {
            if (node->right->balance < 0) {
                rotateleft(node);
            }
            else {
                rotateright(node->right);
                rotateleft(node);
            }
        }
    }
}

void bitree::rotateleft(avlnode* node)
{
    // rotate node left (counterclockwise)
    avlnode* rightchild = node->right;
    // first let's update the balance.
    node->balance = node->balance + 1 - (rightchild->balance < 0 ? rightchild->balance : 0);
    rightchild->balance = rightchild->balance + 1 + (node->balance > 0 ? node->balance : 0);
    if (node->parent == NULL) { // root
        root = rightchild;
        node->right = rightchild->left;
        if (node->right)
            node->right->parent = node;
        root->parent = NULL;
        node->parent = root;
        root->left = node;
    }
    else {
        if (node->parent->left == node)
            node->parent->left = rightchild;
        else
            node->parent->right = rightchild;
        node->right = rightchild->left;
        if (node->right)
            node->right->parent = node;
        rightchild->parent = node->parent;
        node->parent = rightchild;
        rightchild->left = node;
    }
}

void bitree::rotateright(avlnode* node)
{
    // rotate node right (clockwise)
    avlnode* leftchild = node->left;
    node->balance = node->balance - 1 - (leftchild->balance > 0 ? leftchild->balance : 0);
    leftchild->balance = leftchild->balance - 1 + (node->balance < 0 ? node->balance : 0);
    if (node->parent == NULL) { // root
        root = leftchild;
        node->left = leftchild->right;
        if (node->left)
            node->left->parent = node;
        root->parent = NULL;
        node->parent = root;
        root->right = node;
    }
    else {
        if (node->parent->left == node)
            node->parent->left = leftchild;
        else
            node->parent->right = leftchild;
        node->left = leftchild->right;
        if (node->left)
            node->left->parent = node;
        leftchild->parent = node->parent;
        node->parent = leftchild;
        leftchild->right = node;
    }
}

double hvol3d(deque<individual*> P, double cl[DIMENSIONS],
    double fmax[DIMENSIONS])
{
    // Calculate the 3d hypervolume bounded by cl (from below) and fmax (from
    // above).
    bitree tree(cl[0], cl[1]);
    double answer = 0, oldz;
    int i; // must be available outside of loop too.
    if (!P.size() || cl[0] >= fmax[0] || cl[1] >= fmax[1] || cl[2] >= fmax[2])
        return 0;
    // Insert one point at a time to create the horizontal slices. We ignore
    // parts of the hypervolume below cl and chop off parts above fmax.
    if (P[P.size() - 1]->f[0] > cl[0] && P[P.size() - 1]->f[1] > cl[1])
        tree.attemptcandidate(min(fmax[0], P[P.size() - 1]->f[0]),
            min(fmax[1], P[P.size() - 1]->f[1]));
    oldz = P[P.size() - 1]->f[2];
    for (i = P.size() - 2; i >= 0; i--) {
        if (P[i]->f[2] <= cl[2]) { // done, let's calculate the last slice
            break;
        }
        if (P[i]->f[2] < fmax[2]) {
            answer += (min(fmax[2], oldz) - P[i]->f[2]) * tree.totalS;
        }
        if (P[i]->f[0] > cl[0] && P[i]->f[1] > cl[1]) {
            tree.attemptcandidate(min(fmax[0], P[i]->f[0]), min(fmax[1], P[i]->f[1]));
        }
        oldz = P[i]->f[2];
    }
    answer += (min(fmax[2], oldz) - cl[2]) * tree.totalS;
    return answer;
}

// Calculate the 2d hypervolume of a slice of a 3d population.
double calculateslice(deque<individual*> P, double r[DIMENSIONS],
    double fmax[DIMENSIONS], int dimension)
{
    int x, y;
    if (dimension == 0)
        x = 1;
    else
        x = 0;
    if (dimension == 2)
        y = 1;
    else
        y = 2;
    if (fmax[x] <= r[x] || fmax[y] <= r[y])
        return 0;
    bitree tree(r[x], r[y]);
    for (int i = P.size() - 1; i >= 0; i--) {
        if (P[i]->f[dimension] <= fmax[dimension])
            break;
        if (P[i]->f[x] > r[x] && P[i]->f[y] > r[y])
            tree.attemptcandidate(min(fmax[x], P[i]->f[x]), min(fmax[y], P[i]->f[y]));
    }
    return tree.totalS;
}

// Returns the 2d hypervolume for the population P with reference
// point r.
double calculateS(deque<individual*> P, double r[])
{
    double answer = 0;
    sort(P.begin(), P.end(), xcomparator);
    if (P.size()) {
        answer += (P[P.size() - 1]->f[0] - r[0]) * (P[P.size() - 1]->f[1] - r[1]);
        for (int i = P.size() - 2; i >= 0; i--) {
            answer += (P[i]->f[0] - r[0]) * (P[i]->f[1] - P[i + 1]->f[1]);
        }
    }
    return answer;
}
