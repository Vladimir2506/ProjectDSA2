#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

constexpr int nMinIgnoreDepth = 5;
constexpr int nMaxSegmentDepth = 8;
constexpr int nResolution = 300;

enum NodeStatus { UNCHECKED, INTERNAL, INTERSECT, EXTERNAL };
enum PointStatus { UNDISCOVERED, OUT, IN, SURFACE };

template <typename T>
struct Point
{
	T Xpos;
	T Ypos;
	T Zpos;

	Point(T x = 0.0, T y = 0.0, T z = 0.0) :Xpos(x), Ypos(y), Zpos(z) {}

	bool IsInRange() const
	{
		return Xpos >= 0 && Xpos < nResolution && Ypos >= 0 && Ypos < nResolution && Zpos >= 0 && Zpos < nResolution;
	}
};

struct Node {
	vector<Node*> pChildren;
	int nDepth;
	NodeStatus state;

	int Xmax, Xmin, Ymax, Ymin, Zmax, Zmin;

	Node(int x1, int x2, int y1, int y2, int z1, int z2, int nDepth = 0)
		:pChildren(8, nullptr), state(NodeStatus::UNCHECKED),
		Xmax(x1), Xmin(x2), Ymax(y1), Ymin(y2), Zmax(z1), Zmin(z2),
		nDepth(nDepth) {}

	bool Expand();
};

class Model;

struct Octree {
	Model* pModel;
	Node* pRoot;
	Node* SearchPoint(const Point<int>& pt, Node* current);

	void CheckNode(Node* pNode);
	void CheckModel(Node* pNode);

	void GrowOnSurface(Node* pNode);
	void GetSurface(Node* pNode);
	bool MarkSurface(const Point<int>& pt);
};


// 用于判断投影是否在visual hull内部
struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;
	cv::Mat m_image;
	const uint m_threshold = 125;

	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(double x, double y, double z);
};


// 用于index和实际坐标之间的转换
struct CoordinateInfo
{
	int m_resolution;
	double m_min;
	double m_max;

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution), m_min(min), m_max(max) {}
};

class Model
{
public:
	Model(int resX = 100, int resY = 100, int resZ = 100);

	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);

	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	int m_neighbourSize;
	vector<Projection> m_projectionList;
	//The tree and the tensor
	vector<vector<vector<PointStatus>>> m_voxel;
	Octree* pTree;
};

bool Node::Expand() 
{
	//Reach Max Depth of Segmentation
	if (nDepth == nMaxSegmentDepth) return false;

	int Xmid = (Xmax + Xmin) >> 1;
	int Ymid = (Ymax + Ymin) >> 1;
	int Zmid = (Zmax + Zmin) >> 1;

	//Partial Expansion
	if (nDepth == nMaxSegmentDepth - 1) 
	{
		if (Xmax > Xmin)
		{
			if (Ymax > Ymin)
			{
				if (Zmax > Zmin)
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[1] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[2] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
					pChildren[3] = new Node(Xmax, Xmid + 1, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
					pChildren[4] = new Node(Xmid, Xmin, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
					pChildren[5] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
					pChildren[6] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmax, Zmid + 1, nDepth + 1);
					pChildren[7] = new Node(Xmax, Xmid + 1, Ymax, Ymid + 1, Zmax, Zmid + 1, nDepth + 1);
				}
				else
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[1] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[2] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
					pChildren[3] = new Node(Xmax, Xmid + 1, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
				}
			}
			else
			{
				if (Zmax > Zmin)
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[1] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[4] = new Node(Xmid, Xmin, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
					pChildren[5] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
				}
				else
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[1] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
				}
			}
		}
		else
		{
			if (Ymax > Ymin)
			{
				if (Zmax > Zmin)
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[2] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
					pChildren[4] = new Node(Xmid, Xmin, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
					pChildren[6] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmax, Zmid + 1, nDepth + 1);
				}
				else
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[2] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
				}
			}
			else
			{
				if (Zmax > Zmin)
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
					pChildren[4] = new Node(Xmid, Xmin, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
				}
				else
				{
					pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
				}
			}
		}
	}
	//Full Expansion
	else 
	{
		pChildren[0] = new Node(Xmid, Xmin, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
		pChildren[1] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmid, Zmin, nDepth + 1);
		pChildren[2] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
		pChildren[3] = new Node(Xmax, Xmid + 1, Ymax, Ymid + 1, Zmid, Zmin, nDepth + 1);
		pChildren[4] = new Node(Xmid, Xmin, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
		pChildren[5] = new Node(Xmax, Xmid + 1, Ymid, Ymin, Zmax, Zmid + 1, nDepth + 1);
		pChildren[6] = new Node(Xmid, Xmin, Ymax, Ymid + 1, Zmax, Zmid + 1, nDepth + 1);
		pChildren[7] = new Node(Xmax, Xmid + 1, Ymax, Ymid + 1, Zmax, Zmid + 1, nDepth + 1);
	}

	return true;
}


Node* Octree::SearchPoint(const Point<int>& pt, Node* pNode) 
{
	//Octree-Search
	if (!pt.IsInRange() || pNode == nullptr) return nullptr;

	if (pNode->pChildren[0] == nullptr) return pNode;
	if (pNode->state == NodeStatus::INTERNAL || pNode->state == NodeStatus::EXTERNAL) return pNode;

	int Xmid = (pNode->Xmax + pNode->Xmin) >> 1;
	int Ymid = (pNode->Ymax + pNode->Ymin) >> 1;
	int Zmid = (pNode->Zmax + pNode->Zmin) >> 1;

	auto index = ((pt.Zpos > Zmid) << 2) | ((pt.Ypos > Ymid) << 1) | ((pt.Xpos > Xmid) << 0);

	return SearchPoint(pt, pNode->pChildren[index]);
}

void Octree::CheckNode(Node* pNode) 
{
	vector<bool> bInternals(8, true);	
	vector<Point<int>> vecPtMdl(8, Point<int>());
	vector<Point<double>> vecPtCoor(8, Point<double>());

	for (int i = 0; i < 8; ++i)
	{
		vecPtMdl[i].Xpos = (i % 2) ? pNode->Xmax : pNode->Xmin;
		vecPtMdl[i].Ypos = ((i >> 1) % 2) ? pNode->Ymax : pNode->Ymin;
		vecPtMdl[i].Zpos = ((i >> 2) % 2) ? pNode->Zmax : pNode->Zmin;
		vecPtCoor[i].Xpos = pModel->m_corrX.index2coor(vecPtMdl[i].Xpos);
		vecPtCoor[i].Ypos = pModel->m_corrY.index2coor(vecPtMdl[i].Ypos);
		vecPtCoor[i].Zpos = pModel->m_corrZ.index2coor(vecPtMdl[i].Zpos);
	}
	
	for (int i = 0; i < 8; ++i)
	{
		if (pModel->m_voxel[vecPtMdl[i].Xpos][vecPtMdl[i].Ypos][vecPtMdl[i].Zpos] == PointStatus::UNDISCOVERED) 
		{
			for (int j = 0; j < pModel->m_projectionList.size(); ++j)
			{
				if (!bInternals[i]) break;
				else bInternals[i] = pModel->m_projectionList[j].checkRange(vecPtCoor[i].Xpos, vecPtCoor[i].Ypos, vecPtCoor[i].Zpos);
			}
			pModel->m_voxel[vecPtMdl[i].Xpos][vecPtMdl[i].Ypos][vecPtMdl[i].Zpos] = bInternals[i] ? PointStatus::IN : PointStatus::OUT;
		}
		else
		{
			bInternals[i] = (pModel->m_voxel[vecPtMdl[i].Xpos][vecPtMdl[i].Ypos][vecPtMdl[i].Zpos] == PointStatus::OUT) ? false : true;
		}
	}

	//Eight vertecies judgement
	//cnt = 8 => internal
	//cnt = 0 => external
	//else => intersect

	int nInnerCount = 0;
	for (int i = 0;i < 8;i++) nInnerCount += bInternals[i];
	if (nInnerCount == 8) pNode->state = NodeStatus::INTERNAL;
	else if (nInnerCount > 0 || pNode->nDepth <= nMinIgnoreDepth) pNode->state = NodeStatus::INTERSECT;
	else if (nInnerCount == 0 && pNode->nDepth >= nMaxSegmentDepth - 1) pNode->state = NodeStatus::EXTERNAL;
}

void Octree::CheckModel(Node* pNode) 
{
	if (pNode == nullptr) return;
	if (pNode->state == NodeStatus::UNCHECKED) CheckNode(pNode);
	//Continue expand and check
	if (pNode->state == NodeStatus::INTERSECT && 
		(pNode->Xmax > pNode->Xmin || pNode->Ymax > pNode->Ymin || pNode->Zmax > pNode->Zmin)) 
	{
		if (pNode->Expand())
		{
			for (int i = 0; i < 8; ++i)
			{
				CheckModel(pNode->pChildren[i]);
			}
		}
	}
}

void Octree::GrowOnSurface(Node* pNode) 
{
	//Eight Vertecies Judgement
	if (pNode == nullptr) return;
	if (pNode->state == NodeStatus::INTERNAL) 
	{
		if (pNode->nDepth == nMaxSegmentDepth) 
		{
			Point<int> p(pNode->Xmax, pNode->Ymax, pNode->Zmax);
			MarkSurface(p);
		}
		else 
		{
			bool bSurface = false;
			Point<int> ptemp;
			for (int i = 0;i < 8;i++) 
			{
				ptemp.Xpos = (i % 2) ? pNode->Xmax : pNode->Xmin;
				ptemp.Ypos = ((i >> 1) % 2) ? pNode->Ymax : pNode->Ymin;
				ptemp.Zpos = ((i >> 2) % 2) ? pNode->Zmax : pNode->Zmin;

				if (MarkSurface(ptemp)) 
				{
					bSurface = true;
					break;
				}
			}

			if (bSurface) 
			{
				//Try to grow on
				pNode->Expand();
				for (int i = 0;i < 8;i++) 
				{
					if (pNode->pChildren[i]) 
					{
						pNode->pChildren[i]->state = NodeStatus::INTERNAL;
						GrowOnSurface(pNode->pChildren[i]);
					}
				}
			}
		}
	}
}


void Octree::GetSurface(Node* pNode) 
{
	//Find the leaves and grow on surface
	if (pNode == nullptr) return;
	if (pNode->pChildren[0] != nullptr)
	{
		for (int i = 0; i < 8; ++i)
		{
			GetSurface(pNode->pChildren[i]);
		}
	}
	else if (pNode->state == NodeStatus::INTERNAL)
	{
		GrowOnSurface(pNode);
	}
}

bool Octree::MarkSurface(const Point<int>& pt) 
{
	// Discovered surface
	if (pModel->m_voxel[pt.Xpos][pt.Ypos][pt.Zpos] == PointStatus::SURFACE) return true;

	//Six neighbour Search
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };
	auto outOfRange = [&](int indexX, int indexY, int indexZ) 
	{
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= pModel->m_corrX.m_resolution
			|| indexY >= pModel->m_corrY.m_resolution
			|| indexZ >= pModel->m_corrZ.m_resolution;
	};
	bool bSurface = false;

	for (int i = 0; i < 6; i++)
	{
		Point<int> ptNeighbour(pt.Xpos + dx[i], pt.Ypos + dy[i], pt.Zpos + dz[i]);
		//One out leads to surface
		if (outOfRange(ptNeighbour.Xpos, ptNeighbour.Ypos, ptNeighbour.Zpos) ||
			pModel->m_voxel[ptNeighbour.Xpos][ptNeighbour.Ypos][ptNeighbour.Zpos] == PointStatus::OUT)
		{
			bSurface = true;
			break;
		}
		//UNDISCOVERED Points
		else if (pModel->m_voxel[ptNeighbour.Xpos][ptNeighbour.Ypos][ptNeighbour.Zpos] == PointStatus::UNDISCOVERED) 
		{
			auto status = SearchPoint(ptNeighbour,pRoot)->state;
			if (status == EXTERNAL)
			{
				pModel->m_voxel[ptNeighbour.Xpos][ptNeighbour.Ypos][ptNeighbour.Zpos] = PointStatus::OUT;
			}
			else
			{
				pModel->m_voxel[ptNeighbour.Xpos][ptNeighbour.Ypos][ptNeighbour.Zpos] = PointStatus::IN;
			}
			if (status == NodeStatus::EXTERNAL) 
			{
				bSurface = true;
				break;
			}
		}
	}
	if (bSurface) pModel->m_voxel[pt.Xpos][pt.Ypos][pt.Zpos] = PointStatus::SURFACE;
	return bSurface;
}

bool Projection::checkRange(double x, double y, double z)
{
	Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
	int indX = vec3[1] / vec3[2];
	int indY = vec3[0] / vec3[2];

	if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
		return false;
	return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
}

Model::Model(int resX, int resY, int resZ)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)
	, m_voxel(resX, vector<vector<PointStatus>>(resY, vector<PointStatus>(resZ, PointStatus::UNDISCOVERED)))
{
	if (resX > 100)
		m_neighbourSize = resX / 100;
	else
		m_neighbourSize = 3;
}

void Model::saveModel(const char* pFileName)
{
	ofstream fout(pFileName);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_voxel[indexX][indexY][indexZ] == PointStatus::SURFACE)
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
				}
	
}


void Model::saveModelWithNormal(const char* pFileName)
{
	ofstream fout(pFileName);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_voxel[indexX][indexY][indexZ] == PointStatus::SURFACE)
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';

					Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);
					fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
				}
	
}

void Model::loadMatrix(const char* pFileName)
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);
	}
}

void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();
	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	for (int i = 0; i < fileCount; i++)
	{
		m_projectionList[i].m_image = cv::imread(fileName + to_string(i) + pSuffix, CV_8UC1);
	}
}

void Model::getModel()
{
	pTree->CheckModel(pTree->pRoot);
}

void Model::getSurface()
{
	pTree->GetSurface(pTree->pRoot);
}

Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	vector<Eigen::Vector3f> neiborList;
	vector<Eigen::Vector3f> innerList;

	for (int dX = -m_neighbourSize; dX <= m_neighbourSize; dX++)
		for (int dY = -m_neighbourSize; dY <= m_neighbourSize; dY++)
			for (int dZ = -m_neighbourSize; dZ <= m_neighbourSize; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);

					if (m_voxel[neiborX][neiborY][neiborZ] == PointStatus::SURFACE)
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else if (m_voxel[neiborX][neiborY][neiborZ] == PointStatus::IN)
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}

int main()
{
	auto chrT0 = system_clock::now();

	Octree tree;
	Model model(nResolution, nResolution, nResolution);
	Node root(nResolution - 1, 0, nResolution - 1, 0, nResolution - 1, 0);
	tree.pRoot = &root;
	tree.pModel = &model;
	model.pTree = &tree;

	model.loadMatrix("../../calibParamsI.txt");
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");
	auto chrT1 = system_clock::now();
	cout << "准备数据用时：" << duration_cast<milliseconds>(chrT1 - chrT0).count() << "ms" << endl;
	
	model.getModel();
	model.getSurface();
	auto chrT2 = system_clock::now();
	cout << "生成模型用时：" << duration_cast<milliseconds>(chrT2 - chrT1).count() << "ms" << endl;
	
	model.saveModel("../../WithoutNormal.xyz");
	auto chrT3 = system_clock::now();
	cout << "Non-Normal用时：" << duration_cast<milliseconds>(chrT3 - chrT2).count() << "ms" << endl;
	
	model.saveModelWithNormal("../../WithNormal.xyz");
	auto chrT4 = system_clock::now();
	cout << "Normal用时：" << duration_cast<milliseconds>(chrT4 - chrT3).count() << "ms" << endl;

	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	auto chrT5 = system_clock::now();
	cout << "Poisson重建用时：" << duration_cast<milliseconds>(chrT5 - chrT4).count() << "ms" << endl;

	cout << "整体用时：" << duration_cast<milliseconds>(chrT5 - chrT0).count() << "ms" << endl;
	
	return 0;
}