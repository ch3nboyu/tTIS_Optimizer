# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'win_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 705)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        MainWindow.setFont(font)
        # centralwidget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        # 创建一个滚动区域
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 331, 1298))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setObjectName("label")
        self.verticalLayout_5.addWidget(self.label)
        self.data_path_edit = QtWidgets.QLineEdit(self.groupBox_4)
        self.data_path_edit.setObjectName("data_path_edit")
        self.verticalLayout_5.addWidget(self.data_path_edit)
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_5.addWidget(self.label_2)
        self.output_path_edit = QtWidgets.QLineEdit(self.groupBox_4)
        self.output_path_edit.setObjectName("output_path_edit")
        self.verticalLayout_5.addWidget(self.output_path_edit)
        self.verticalLayout.addWidget(self.groupBox_4)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.subMark_edit = QtWidgets.QLineEdit(self.widget)
        self.subMark_edit.setObjectName("subMark_edit")
        self.verticalLayout_4.addWidget(self.subMark_edit)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.simMark_edit = QtWidgets.QLineEdit(self.widget)
        self.simMark_edit.setObjectName("simMark_edit")
        self.verticalLayout_4.addWidget(self.simMark_edit)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)
        self.exampleIdx_edit = QtWidgets.QLineEdit(self.widget)
        self.exampleIdx_edit.setObjectName("exampleIdx_edit")
        self.verticalLayout_4.addWidget(self.exampleIdx_edit)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_4.addWidget(self.label_6)
        self.eeg_posi_edit = QtWidgets.QLineEdit(self.widget)
        self.eeg_posi_edit.setObjectName("eeg_posi_edit")
        self.verticalLayout_4.addWidget(self.eeg_posi_edit)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_6.addWidget(self.label_11, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.widget)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 1, 1, 1)
        self.initialElecNum_edit = QtWidgets.QLineEdit(self.widget)
        self.initialElecNum_edit.setObjectName("initialElecNum_edit")
        self.gridLayout_6.addWidget(self.initialElecNum_edit, 1, 0, 1, 1)
        self.cur_max_edit = QtWidgets.QLineEdit(self.widget)
        self.cur_max_edit.setObjectName("cur_max_edit")
        self.gridLayout_6.addWidget(self.cur_max_edit, 1, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 2, 0, 1, 1)
        self.label_52 = QtWidgets.QLabel(self.widget)
        self.label_52.setObjectName("label_52")
        self.gridLayout_6.addWidget(self.label_52, 2, 1, 1, 1)
        self.cur_min_edit = QtWidgets.QLineEdit(self.widget)
        self.cur_min_edit.setObjectName("cur_min_edit")
        self.gridLayout_6.addWidget(self.cur_min_edit, 3, 0, 1, 1)
        self.cur_sum_edit = QtWidgets.QLineEdit(self.widget)
        self.cur_sum_edit.setObjectName("cur_sum_edit")
        self.gridLayout_6.addWidget(self.cur_sum_edit, 3, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.widget)
        self.label_15.setObjectName("label_15")
        self.gridLayout_6.addWidget(self.label_15, 4, 0, 1, 1)
        self.label_53 = QtWidgets.QLabel(self.widget)
        self.label_53.setObjectName("label_53")
        self.gridLayout_6.addWidget(self.label_53, 4, 1, 1, 1)
        self.precision_edit = QtWidgets.QLineEdit(self.widget)
        self.precision_edit.setObjectName("precision_edit")
        self.gridLayout_6.addWidget(self.precision_edit, 5, 0, 1, 1)
        self.type_edit = QtWidgets.QComboBox(self.widget)
        self.type_edit.setObjectName("type_edit")
        self.type_edit.addItem("")
        self.type_edit.addItem("")
        self.gridLayout_6.addWidget(self.type_edit, 5, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 6, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setObjectName("label_9")
        self.gridLayout_6.addWidget(self.label_9, 6, 1, 1, 1)
        self.thres_edit = QtWidgets.QLineEdit(self.widget)
        self.thres_edit.setObjectName("thres_edit")
        self.gridLayout_6.addWidget(self.thres_edit, 7, 0, 1, 1)
        self.nt_edit = QtWidgets.QLineEdit(self.widget)
        self.nt_edit.setObjectName("nt_edit")
        self.gridLayout_6.addWidget(self.nt_edit, 7, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_6.addWidget(self.label_14, 8, 0, 1, 1)
        self.method_combo = QtWidgets.QComboBox(self.widget)
        self.method_combo.setObjectName("method_combo")
        self.method_combo.addItem("")
        self.method_combo.addItem("")
        self.gridLayout_6.addWidget(self.method_combo, 9, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_6)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.widget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.elit_ratio_edit = QtWidgets.QLineEdit(self.tab_3)
        self.elit_ratio_edit.setObjectName("elit_ratio_edit")
        self.gridLayout_7.addWidget(self.elit_ratio_edit, 3, 1, 1, 1)
        self.label_48 = QtWidgets.QLabel(self.tab_3)
        self.label_48.setObjectName("label_48")
        self.gridLayout_7.addWidget(self.label_48, 0, 0, 1, 1)
        self.label_47 = QtWidgets.QLabel(self.tab_3)
        self.label_47.setObjectName("label_47")
        self.gridLayout_7.addWidget(self.label_47, 6, 0, 1, 1)
        self.cross_prob_edit = QtWidgets.QLineEdit(self.tab_3)
        self.cross_prob_edit.setObjectName("cross_prob_edit")
        self.gridLayout_7.addWidget(self.cross_prob_edit, 5, 0, 1, 1)
        self.label_58 = QtWidgets.QLabel(self.tab_3)
        self.label_58.setObjectName("label_58")
        self.gridLayout_7.addWidget(self.label_58, 0, 1, 1, 1)
        self.label_49 = QtWidgets.QLabel(self.tab_3)
        self.label_49.setObjectName("label_49")
        self.gridLayout_7.addWidget(self.label_49, 2, 0, 1, 1)
        self.label_56 = QtWidgets.QLabel(self.tab_3)
        self.label_56.setObjectName("label_56")
        self.gridLayout_7.addWidget(self.label_56, 2, 1, 1, 1)
        self.label_57 = QtWidgets.QLabel(self.tab_3)
        self.label_57.setObjectName("label_57")
        self.gridLayout_7.addWidget(self.label_57, 4, 1, 1, 1)
        self.parents_portion_edit = QtWidgets.QLineEdit(self.tab_3)
        self.parents_portion_edit.setObjectName("parents_portion_edit")
        self.gridLayout_7.addWidget(self.parents_portion_edit, 5, 1, 1, 1)
        self.label_54 = QtWidgets.QLabel(self.tab_3)
        self.label_54.setObjectName("label_54")
        self.gridLayout_7.addWidget(self.label_54, 6, 1, 1, 1)
        self.pop_size_edit = QtWidgets.QLineEdit(self.tab_3)
        self.pop_size_edit.setObjectName("pop_size_edit")
        self.gridLayout_7.addWidget(self.pop_size_edit, 1, 1, 1, 1)
        # self.time_threshold_edit = QtWidgets.QLineEdit(self.tab_3)
        # self.time_threshold_edit.setObjectName("time_threshold_edit")
        # self.gridLayout_7.addWidget(self.time_threshold_edit, 7, 1, 1, 1)
        self.mut_prob_edit = QtWidgets.QLineEdit(self.tab_3)
        self.mut_prob_edit.setObjectName("mut_prob_edit")
        self.gridLayout_7.addWidget(self.mut_prob_edit, 3, 0, 1, 1)
        self.label_55 = QtWidgets.QLabel(self.tab_3)
        self.label_55.setObjectName("label_55")
        self.gridLayout_7.addWidget(self.label_55, 4, 0, 1, 1)
        # self.cross_type_edit = QtWidgets.QLineEdit(self.tab_3)
        # self.cross_type_edit.setObjectName("cross_type_edit")
        # self.gridLayout_7.addWidget(self.cross_type_edit, 7, 0, 1, 1)
        self.max_iter_edit = QtWidgets.QLineEdit(self.tab_3)
        self.max_iter_edit.setObjectName("max_iter_edit")
        self.gridLayout_7.addWidget(self.max_iter_edit, 1, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_22 = QtWidgets.QLabel(self.tab_4)
        self.label_22.setObjectName("label_22")
        self.gridLayout_5.addWidget(self.label_22, 0, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.tab_4)
        self.label_23.setObjectName("label_23")
        self.gridLayout_5.addWidget(self.label_23, 0, 1, 1, 1)
        self.para_1_edit = QtWidgets.QLineEdit(self.tab_4)
        self.para_1_edit.setObjectName("para_1_edit")
        self.gridLayout_5.addWidget(self.para_1_edit, 1, 0, 1, 1)
        self.para_2_edit = QtWidgets.QLineEdit(self.tab_4)
        self.para_2_edit.setObjectName("para_2_edit")
        self.gridLayout_5.addWidget(self.para_2_edit, 1, 1, 1, 1)
        self.label_44 = QtWidgets.QLabel(self.tab_4)
        self.label_44.setObjectName("label_44")
        self.gridLayout_5.addWidget(self.label_44, 2, 0, 1, 1)
        self.label_45 = QtWidgets.QLabel(self.tab_4)
        self.label_45.setObjectName("label_45")
        self.gridLayout_5.addWidget(self.label_45, 2, 1, 1, 1)
        self.para_3_edit = QtWidgets.QLineEdit(self.tab_4)
        self.para_3_edit.setObjectName("para_3_edit")
        self.gridLayout_5.addWidget(self.para_3_edit, 3, 0, 1, 1)
        self.para_4_edit = QtWidgets.QLineEdit(self.tab_4)
        self.para_4_edit.setObjectName("para_4_edit")
        self.gridLayout_5.addWidget(self.para_4_edit, 3, 1, 1, 1)
        self.label_46 = QtWidgets.QLabel(self.tab_4)
        self.label_46.setObjectName("label_46")
        self.gridLayout_5.addWidget(self.label_46, 4, 0, 1, 1)
        self.para_5_edit = QtWidgets.QLineEdit(self.tab_4)
        self.para_5_edit.setObjectName("para_5_edit")
        self.gridLayout_5.addWidget(self.para_5_edit, 5, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_4, "")
        self.verticalLayout.addWidget(self.tabWidget_2)
        self.tabWidget_case = QtWidgets.QTabWidget(self.widget)
        self.tabWidget_case.setObjectName("tabWidget_case")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.case1_num_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_num_edit.setObjectName("case1_num_edit")
        self.gridLayout_2.addWidget(self.case1_num_edit, 0, 1, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.tab)
        self.label_26.setObjectName("label_26")
        self.gridLayout_2.addWidget(self.label_26, 6, 0, 1, 1)
        self.case1_shape_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_shape_edit.setObjectName("case1_shape_edit")
        self.gridLayout_2.addWidget(self.case1_shape_edit, 3, 1, 1, 1)
        self.case1_coordMNI_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_coordMNI_edit.setObjectName("case1_coordMNI_edit")
        self.gridLayout_2.addWidget(self.case1_coordMNI_edit, 2, 1, 1, 1)
        self.case1_radius_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_radius_edit.setObjectName("case1_radius_edit")
        self.gridLayout_2.addWidget(self.case1_radius_edit, 4, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.tab)
        self.label_30.setObjectName("label_30")
        self.gridLayout_2.addWidget(self.label_30, 3, 0, 1, 1)
        self.case1_intensity_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_intensity_edit.setObjectName("case1_intensity_edit")
        self.gridLayout_2.addWidget(self.case1_intensity_edit, 5, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.tab)
        self.label_29.setObjectName("label_29")
        self.gridLayout_2.addWidget(self.label_29, 4, 0, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.tab)
        self.label_27.setObjectName("label_27")
        self.gridLayout_2.addWidget(self.label_27, 5, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.tab)
        self.label_25.setObjectName("label_25")
        self.gridLayout_2.addWidget(self.label_25, 0, 0, 1, 1)
        self.case1_otheralpha_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_otheralpha_edit.setObjectName("case1_otheralpha_edit")
        self.gridLayout_2.addWidget(self.case1_otheralpha_edit, 6, 1, 1, 1)
        self.case1_name_edit = QtWidgets.QLineEdit(self.tab)
        self.case1_name_edit.setObjectName("case1_name_edit")
        self.gridLayout_2.addWidget(self.case1_name_edit, 1, 1, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.tab)
        self.label_28.setObjectName("label_28")
        self.gridLayout_2.addWidget(self.label_28, 2, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.tab)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 1, 0, 1, 1)
        self.tabWidget_case.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_31 = QtWidgets.QLabel(self.tab_2)
        self.label_31.setObjectName("label_31")
        self.gridLayout_3.addWidget(self.label_31, 0, 0, 1, 1)
        self.case2_num_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_num_edit.setObjectName("case2_num_edit")
        self.gridLayout_3.addWidget(self.case2_num_edit, 0, 1, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.tab_2)
        self.label_36.setObjectName("label_36")
        self.gridLayout_3.addWidget(self.label_36, 1, 0, 1, 1)
        self.case2_name_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_name_edit.setObjectName("case2_name_edit")
        self.gridLayout_3.addWidget(self.case2_name_edit, 1, 1, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.tab_2)
        self.label_35.setObjectName("label_35")
        self.gridLayout_3.addWidget(self.label_35, 2, 0, 1, 1)
        self.case2_coordMNI_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_coordMNI_edit.setObjectName("case2_coordMNI_edit")
        self.gridLayout_3.addWidget(self.case2_coordMNI_edit, 2, 1, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.tab_2)
        self.label_37.setObjectName("label_37")
        self.gridLayout_3.addWidget(self.label_37, 3, 0, 1, 1)
        self.case2_shape_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_shape_edit.setObjectName("case2_shape_edit")
        self.gridLayout_3.addWidget(self.case2_shape_edit, 3, 1, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.tab_2)
        self.label_33.setObjectName("label_33")
        self.gridLayout_3.addWidget(self.label_33, 4, 0, 1, 1)
        self.case2_radius_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_radius_edit.setObjectName("case2_radius_edit")
        self.gridLayout_3.addWidget(self.case2_radius_edit, 4, 1, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.tab_2)
        self.label_32.setObjectName("label_32")
        self.gridLayout_3.addWidget(self.label_32, 5, 0, 1, 1)
        self.case2_intensity_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_intensity_edit.setObjectName("case2_intensity_edit")
        self.gridLayout_3.addWidget(self.case2_intensity_edit, 5, 1, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.tab_2)
        self.label_34.setObjectName("label_34")
        self.gridLayout_3.addWidget(self.label_34, 6, 0, 1, 1)
        self.case2_otheralpha_edit = QtWidgets.QLineEdit(self.tab_2)
        self.case2_otheralpha_edit.setObjectName("case2_otheralpha_edit")
        self.gridLayout_3.addWidget(self.case2_otheralpha_edit, 6, 1, 1, 1)
        self.tabWidget_case.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget_case)
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_42 = QtWidgets.QLabel(self.groupBox)
        self.label_42.setObjectName("label_42")
        self.gridLayout.addWidget(self.label_42, 0, 0, 1, 1)
        self.avoid_num_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_num_edit.setObjectName("avoid_num_edit")
        self.gridLayout.addWidget(self.avoid_num_edit, 0, 1, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.groupBox)
        self.label_39.setObjectName("label_39")
        self.gridLayout.addWidget(self.label_39, 1, 0, 1, 1)
        self.avoid_name_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_name_edit.setObjectName("avoid_name_edit")
        self.gridLayout.addWidget(self.avoid_name_edit, 1, 1, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.groupBox)
        self.label_38.setObjectName("label_38")
        self.gridLayout.addWidget(self.label_38, 2, 0, 1, 1)
        self.avoid_coordMNI_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_coordMNI_edit.setObjectName("avoid_coordMNI_edit")
        self.gridLayout.addWidget(self.avoid_coordMNI_edit, 2, 1, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.groupBox)
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 3, 0, 1, 1)
        self.avoid_shape_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_shape_edit.setObjectName("avoid_shape_edit")
        self.gridLayout.addWidget(self.avoid_shape_edit, 3, 1, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.groupBox)
        self.label_41.setObjectName("label_41")
        self.gridLayout.addWidget(self.label_41, 4, 0, 1, 1)
        self.avoid_radius_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_radius_edit.setObjectName("avoid_radius_edit")
        self.gridLayout.addWidget(self.avoid_radius_edit, 4, 1, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.groupBox)
        self.label_43.setObjectName("label_43")
        self.gridLayout.addWidget(self.label_43, 5, 0, 1, 1)
        self.avoid_coef_edit = QtWidgets.QLineEdit(self.groupBox)
        self.avoid_coef_edit.setObjectName("avoid_coef_edit")
        self.gridLayout.addWidget(self.avoid_coef_edit, 5, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.verticalLayout_2.addWidget(self.widget)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.addWidget(self.scrollArea)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.save_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_button.setObjectName("save_button")
        self.horizontalLayout.addWidget(self.save_button)
        self.prepare_button = QtWidgets.QPushButton(self.centralwidget)
        self.prepare_button.setObjectName("prepare_button")
        self.horizontalLayout.addWidget(self.prepare_button)
        self.optimization_button = QtWidgets.QPushButton(self.centralwidget)
        self.optimization_button.setObjectName("optimization_button")
        self.horizontalLayout.addWidget(self.optimization_button)
        self.simulation_button = QtWidgets.QPushButton(self.centralwidget)
        self.simulation_button.setObjectName("simulation_button")
        self.horizontalLayout.addWidget(self.simulation_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.right_widget = QtWidgets.QTextEdit(self.centralwidget)
        self.right_widget.setReadOnly(True)
        self.right_widget.setObjectName("right_widget")
        self.verticalLayout_3.addWidget(self.right_widget)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.horizontalLayout_2.setStretch(1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget_case.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "tTIS Optimizer"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Path Parameter"))
        self.label.setText(_translate("MainWindow", "data_path"))
        self.label_2.setText(_translate("MainWindow", "output_path"))
        self.label_3.setText(_translate("MainWindow", "subMark"))
        self.label_4.setText(_translate("MainWindow", "simMark"))
        self.label_5.setText(_translate("MainWindow", "exampleIdx"))
        self.label_6.setText(_translate("MainWindow", "eeg-posi"))
        self.label_11.setText(_translate("MainWindow", "initialElecNum"))
        self.label_12.setText(_translate("MainWindow", "cur_max"))
        self.label_10.setText(_translate("MainWindow", "cur_min"))
        self.label_52.setText(_translate("MainWindow", "cur_sum"))
        self.label_15.setText(_translate("MainWindow", "precision"))
        self.label_53.setText(_translate("MainWindow", "type"))
        self.type_edit.setItemText(0, _translate("MainWindow", "tet"))
        self.type_edit.setItemText(1, _translate("MainWindow", "tri"))
        self.label_8.setText(_translate("MainWindow", "thres"))
        self.label_9.setText(_translate("MainWindow", "nt"))
        self.label_14.setText(_translate("MainWindow", "Method"))
        self.method_combo.setItemText(0, _translate("MainWindow", "GeneticAlgorithm"))
        self.method_combo.setItemText(1, _translate("MainWindow", "ParticleSwarm"))
        self.label_48.setText(_translate("MainWindow", "max_iter"))
        # self.label_47.setText(_translate("MainWindow", "cross_type"))
        self.label_58.setText(_translate("MainWindow", "pop_size"))
        self.label_49.setText(_translate("MainWindow", "mut_prob"))
        self.label_56.setText(_translate("MainWindow", "elit_ratio"))
        self.label_57.setText(_translate("MainWindow", "parents_portion"))
        # self.label_54.setText(_translate("MainWindow", "time_threshold"))
        self.label_55.setText(_translate("MainWindow", "cross_prob"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "GA_para"))
        self.label_22.setText(_translate("MainWindow", "para_1"))
        self.label_23.setText(_translate("MainWindow", "para_2"))
        self.label_44.setText(_translate("MainWindow", "para_3"))
        self.label_45.setText(_translate("MainWindow", "para_4"))
        self.label_46.setText(_translate("MainWindow", "para_5"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "PSO_para"))
        self.label_26.setText(_translate("MainWindow", "otheralpha"))
        self.label_30.setText(_translate("MainWindow", "shape"))
        self.label_29.setText(_translate("MainWindow", "radius"))
        self.label_27.setText(_translate("MainWindow", "intensity"))
        self.label_25.setText(_translate("MainWindow", "num"))
        self.label_28.setText(_translate("MainWindow", "coordMNI"))
        self.label_17.setText(_translate("MainWindow", "name"))
        self.tabWidget_case.setTabText(self.tabWidget_case.indexOf(self.tab), _translate("MainWindow", "case1"))
        self.label_31.setText(_translate("MainWindow", "num"))
        self.label_36.setText(_translate("MainWindow", "name"))
        self.label_35.setText(_translate("MainWindow", "coordMNI"))
        self.label_37.setText(_translate("MainWindow", "shape"))
        self.label_33.setText(_translate("MainWindow", "radius"))
        self.label_32.setText(_translate("MainWindow", "intensity"))
        self.label_34.setText(_translate("MainWindow", "otheralpha"))
        self.tabWidget_case.setTabText(self.tabWidget_case.indexOf(self.tab_2), _translate("MainWindow", "case2"))
        self.groupBox.setTitle(_translate("MainWindow", "avoid"))
        self.label_42.setText(_translate("MainWindow", "num"))
        self.label_39.setText(_translate("MainWindow", "name"))
        self.label_38.setText(_translate("MainWindow", "coordMNI"))
        self.label_40.setText(_translate("MainWindow", "shape"))
        self.label_41.setText(_translate("MainWindow", "radius"))
        self.label_43.setText(_translate("MainWindow", "coef"))
        self.save_button.setText(_translate("MainWindow", "Save Config"))
        self.prepare_button.setText(_translate("MainWindow", "Prepare"))
        self.optimization_button.setText(_translate("MainWindow", "Optimization"))
        self.simulation_button.setText(_translate("MainWindow", "Simulation"))
