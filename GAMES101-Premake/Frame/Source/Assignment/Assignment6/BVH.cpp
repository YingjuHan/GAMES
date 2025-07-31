#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
    SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
    primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* pNode = new BVHBuildNode();

    // ��������İ�Χ�м���
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
    {
        bounds = Union(bounds, objects[i]->getBounds());
    }

    // ֻ����һ�����壬����Ҷ�ڵ�
    if (objects.size() == 1)
    {
        pNode->bounds = objects[0]->getBounds();
        pNode->object = objects[0];
        pNode->left = nullptr;
        pNode->right = nullptr;

        return pNode;
    }
    // �����������壬������ left �� right Ҷ�ڵ�
    else if (objects.size() == 2)
    {
        pNode->left = recursiveBuild(std::vector{ objects[0] });
        pNode->right = recursiveBuild(std::vector{ objects[1] });
        pNode->bounds = Union(pNode->left->bounds, pNode->right->bounds);

        return pNode;
    }
    else
    {
        // �����������������ɵİ�Χ��
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
        {
            centroidBounds = Union(centroidBounds, objects[i]->getBounds().Centroid());
        }

        // ���ά��
        switch (centroidBounds.maxExtent())
        {
        case 0:
            // X ��
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                {
                    // ���������尴�����Χ�е����ĵ� X ������Y ��� Z ��� case ͬ��
                    return f1->getBounds().Centroid().x < f2->getBounds().Centroid().x;
                });
            break;
        case 1:
            // Y ��
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                {
                    return f1->getBounds().Centroid().y < f2->getBounds().Centroid().y;
                });
            break;
        case 2:
            // Z ��
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                {
                    return f1->getBounds().Centroid().z < f2->getBounds().Centroid().z;
                });
            break;
        }

        const auto& begin = objects.begin();
        const auto& end = objects.end();

#define ENABLE_SAH 1

#if ENABLE_SAH

        // һ�ݱȽ������� SAH ���ܣ�https://zhuanlan.zhihu.com/p/50720158

        // ���ַ�ʽ������
        constexpr uint8_t SlashCount = 8;
        constexpr float SlashCountInv = 1.0f / (float)SlashCount;
        const float SC = centroidBounds.SurfaceArea();

        // ���ڼ�¼���ŵĻ��ַ�ʽ
        uint8_t minCostIndex = SlashCount / 2;
        float minCost = std::numeric_limits<float>::infinity();

        for (uint8_t index = 1; index < SlashCount; ++index)
        {
            const auto& target = objects.begin() + (objects.size() * index * SlashCountInv);
            auto leftObjects = std::vector<Object*>(begin, target);
            auto rightObjects = std::vector<Object*>(target, end);

            // �ֱ���㻮��֮�������ְ�Χ�еı����
            Bounds3 leftBounds, rightBounds;
            for (const auto& obj : leftObjects)
            {
                leftBounds = Union(leftBounds, obj->getBounds().Centroid());
            }
            for (const auto& obj : rightObjects)
            {
                rightBounds = Union(rightBounds, obj->getBounds().Centroid());
            }

            float SA = leftBounds.SurfaceArea();
            float SB = rightBounds.SurfaceArea();
            float a = leftObjects.size();
            float b = rightObjects.size();
            float cost = (SA * a + SB * b) / SC + 0.125f;

            if (cost < minCost)
            {
                // ���¸��ŵĻ��ַ�ʽ
                minCost = cost;
                minCostIndex = index;
            }
        }

        const auto& target = objects.begin() + (objects.size() * minCostIndex * SlashCountInv);

#else // ENABLE_SAH

        // ������ BVH ���ַ�ʽ�����������м�һ��Ϊ��
        const auto& target = objects.begin() + (objects.size() / 2);

#endif // ENABLE_SAH

        auto leftObjects = std::vector<Object*>(begin, target);
        auto rightObjects = std::vector<Object*>(target, end);

        pNode->left = recursiveBuild(leftObjects);
        pNode->right = recursiveBuild(rightObjects);
        pNode->bounds = Union(pNode->left->bounds, pNode->right->bounds);
    }

    return pNode;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    std::array<int, 3> dirIsNeg{
        (int)(ray.direction.x < 0.0f),
        (int)(ray.direction.y < 0.0f),
        (int)(ray.direction.z < 0.0f)
    };
    if (!node->bounds.IntersectP(ray, ray.direction_inv, std::move(dirIsNeg)))
    {
        return Intersection{};
    }

    // Ҷ�ڵ�
    if (node->left == nullptr && node->right == nullptr)
    {
        return node->object->getIntersection(ray);
    }

    Intersection leaf1 = getIntersection(node->left, ray);
    Intersection leaf2 = getIntersection(node->right, ray);

    return leaf1.distance < leaf2.distance ? leaf1 : leaf2;
}
